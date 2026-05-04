"""Telegram orchestrator that delegates each user message to `claude -p`.

Architecture (Phase 1):
    Telegram update -> bot.py -> subprocess(claude -p) -> stdout -> Telegram reply
    All persona/memory logic lives in CLAUDE.md and the memory/ directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
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
MEMORY_DIR = PROJECT_ROOT / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
RECENT_PATH = MEMORY_DIR / "recent.jsonl"

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
        "Follow CLAUDE.md exactly: read memory/ for context, decide your reply, "
        "do compression / facts.md updates if warranted, then — as the very LAST "
        "thing — output the reply as plain text. Do NOT write to "
        "memory/recent.jsonl yourself; the orchestrator records each turn after "
        "Telegram confirms delivery. Do NOT end on a tool call; the last "
        "plain-text message is what Telegram receives. No preamble, no code "
        "fences, no meta-commentary.\n\n"
        f"<user_message>\n{user_text}\n</user_message>"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_turn(user_text: str, user_ts: str, assistant_text: str, assistant_ts: str) -> None:
    """Append the user/assistant pair to memory/recent.jsonl. Called only after
    Telegram has accepted the reply, so what's in memory always matches what the
    user saw. The per-chat lock + the fact that claude has already exited means
    no concurrent writer can race us."""
    lines = [
        json.dumps({"role": "user", "content": user_text, "ts": user_ts}, ensure_ascii=False),
        json.dumps({"role": "assistant", "content": assistant_text, "ts": assistant_ts}, ensure_ascii=False),
    ]
    payload = "\n".join(lines) + "\n"
    # Defend against a prior writer (e.g. claude compression) leaving no trailing \n.
    if RECENT_PATH.exists() and RECENT_PATH.stat().st_size > 0:
        with RECENT_PATH.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                payload = "\n" + payload
    with RECENT_PATH.open("a", encoding="utf-8") as f:
        f.write(payload)


VAULT_DIR = "/home/xchen/Documents/sync'd"


def _build_args(prompt: str) -> list[str]:
    args = [
        CLAUDE_BIN,
        "-p",
        "--allowedTools",
        "Read,Write,Edit,Glob,Grep,WebFetch,WebSearch",
        "--permission-mode",
        "bypassPermissions",
        "--add-dir",
        VAULT_DIR,
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
    user_ts = _now_iso()
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
        _append_turn(text, user_ts, reply, _now_iso())
        log.info("sent chat=%s reply_len=%d", chat_id, len(reply))


async def _send_with_retry(msg, text: str, attempts: int = 4) -> None:
    """Telegram API can hiccup; retry with backoff 1s, 3s, 9s. If all attempts
    fail we propagate, the outer handler logs, and the turn is NOT appended to
    memory/recent.jsonl — keeping memory consistent with what the user saw."""
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
