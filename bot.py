"""Telegram orchestrator that delegates each user message to `claude -p`.

Architecture (Phase 1):
    Telegram update -> bot.py -> subprocess(claude -p) -> stdout -> Telegram reply
    All persona/memory logic lives in CLAUDE.md and the memory/ directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

PROJECT_ROOT = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
(PROJECT_ROOT / "memory").mkdir(exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "").strip()
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "120"))
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude").strip()

_raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: set[int] = {
    int(x) for x in _raw_ids.replace(",", " ").split() if x.strip().lstrip("-").isdigit()
}

TELEGRAM_MSG_LIMIT = 4000  # leave headroom under 4096
ERROR_REPLY = "我休息一下，等会儿再聊。"
TIMEOUT_REPLY = "我有点慢半拍，等会儿再来问我？"
EMPTY_REPLY = "（一时没话说，再说一遍？）"

handler = RotatingFileHandler(
    LOG_DIR / "bot.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[handler, logging.StreamHandler()],
)
# httpx INFO logs every request URL — for python-telegram-bot that includes the
# bot token in the path. Keep it at WARNING so tokens don't land in logs/.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("tg-companion")

_chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def _is_authorized(user_id: int | None) -> bool:
    if not ALLOWED_USER_IDS:
        return True  # whitelist disabled
    return user_id in ALLOWED_USER_IDS


def _build_prompt(user_text: str) -> str:
    return (
        "A Telegram user just sent you the message in <user_message>. "
        "Follow CLAUDE.md exactly: (1) Read memory/ for context, (2) decide your "
        "reply in your head, (3) Write the new turn into memory/recent.jsonl, "
        "then (4) — as the very LAST thing you do — output the reply as plain "
        "text. Do NOT end on a tool call; the last plain-text message is what "
        "Telegram receives. No preamble, no code fences, no meta-commentary.\n\n"
        f"<user_message>\n{user_text}\n</user_message>"
    )


def _build_args(prompt: str) -> list[str]:
    args = [
        CLAUDE_BIN,
        "-p",
        "--allowedTools",
        "Read,Write",
        "--permission-mode",
        "acceptEdits",
        "--output-format",
        "text",
    ]
    if CLAUDE_MODEL:
        args.extend(["--model", CLAUDE_MODEL])
    args.append(prompt)
    return args


async def _typing_loop(bot, chat_id: int, stop: asyncio.Event) -> None:
    """Keep the typing indicator alive (Telegram's lasts ~5s) until stop is set."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception as e:  # network blip, ignore
            log.debug("typing action failed: %s", e)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue


async def _call_claude(user_text: str) -> str:
    """Invoke `claude -p` and return its stdout. Raises on timeout / non-zero exit."""
    args = _build_args(_build_prompt(user_text))
    log.info("spawn claude pid=? cwd=%s model=%s", PROJECT_ROOT, CLAUDE_MODEL or "(default)")
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=CLAUDE_TIMEOUT
        )
    except asyncio.TimeoutError:
        log.warning("claude pid=%s timed out after %ds; killing", proc.pid, CLAUDE_TIMEOUT)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        raise

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"claude exited {proc.returncode}: {err[:500] or '<no stderr>'}"
        )
    return stdout.decode(errors="replace").strip()


def _chunks(s: str, size: int):
    for i in range(0, len(s), size):
        yield s[i : i + size]


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_authorized(user.id if user else None):
        log.warning("rejected /start from user_id=%s", user.id if user else None)
        return
    await update.message.reply_text(
        "Hi. 在的。直接发消息就行，我会回。"
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.message
    if user is None or msg is None or msg.text is None:
        return

    if not _is_authorized(user.id):
        log.warning(
            "rejected message user_id=%s username=%s text=%r",
            user.id, user.username, msg.text[:60],
        )
        return

    chat_id = msg.chat_id
    text = msg.text
    log.info("recv chat=%s user=%s len=%d", chat_id, user.id, len(text))

    async with _chat_locks[chat_id]:
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_typing_loop(context.bot, chat_id, stop_typing))
        try:
            try:
                reply = await _call_claude(text)
            except asyncio.TimeoutError:
                log.error("claude timeout chat=%s", chat_id)
                await msg.reply_text(TIMEOUT_REPLY)
                return
            except Exception:
                log.exception("claude failure chat=%s", chat_id)
                await msg.reply_text(ERROR_REPLY)
                return
        finally:
            stop_typing.set()
            try:
                await typing_task
            except Exception:
                pass

        if not reply:
            log.warning("claude returned empty reply chat=%s", chat_id)
            await msg.reply_text(EMPTY_REPLY)
            return

        for chunk in _chunks(reply, TELEGRAM_MSG_LIMIT):
            await _send_with_retry(msg, chunk)
        log.info("sent chat=%s reply_len=%d", chat_id, len(reply))


async def _send_with_retry(msg, text: str, attempts: int = 4) -> None:
    """Telegram API can hiccup; reply is already saved in memory/recent.jsonl,
    so it's fine to keep trying. Backoff: 1s, 3s, 9s."""
    delay = 1.0
    for i in range(1, attempts + 1):
        try:
            await msg.reply_text(text)
            if i > 1:
                log.info("reply_text succeeded on attempt %d", i)
            return
        except (TimedOut, NetworkError) as e:
            if i == attempts:
                log.error("reply_text failed after %d attempts: %s", i, e)
                raise
            log.warning("reply_text attempt %d/%d failed (%s); retrying in %.1fs", i, attempts, e, delay)
            await asyncio.sleep(delay)
            delay *= 3


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("unhandled telegram error: %s", context.error)


def main() -> None:
    if not TG_BOT_TOKEN:
        raise SystemExit("TG_BOT_TOKEN is required (set it in .env)")
    if not ALLOWED_USER_IDS:
        log.warning(
            "ALLOWED_USER_IDS is empty — bot will respond to ANYONE. "
            "Set ALLOWED_USER_IDS in .env to lock it down."
        )
    else:
        log.info("whitelist active: %s", sorted(ALLOWED_USER_IDS))

    log.info(
        "starting bot project=%s model=%s timeout=%ds",
        PROJECT_ROOT, CLAUDE_MODEL or "(default)", CLAUDE_TIMEOUT,
    )

    app = (
        Application.builder()
        .token(TG_BOT_TOKEN)
        .connect_timeout(15.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .pool_timeout(10.0)
        .get_updates_read_timeout(40.0)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
