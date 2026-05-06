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
import time
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
MEDIA_DIR = MEMORY_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)
RECENT_PATH = MEMORY_DIR / "recent.jsonl"
FACTS_PATH = MEMORY_DIR / "facts.md"
SUMMARY_PATH = MEMORY_DIR / "summary.md"
RECENT_TAIL_LINES = 30  # how many recent.jsonl lines to inject into the prompt
ALBUM_DEBOUNCE_S = 1.5  # wait this long after the latest media-group photo before flushing

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
# Buffers media-group photos until the group goes quiet for ALBUM_DEBOUNCE_S so
# we can fire a single Cas turn over the whole album rather than one per photo.
_album_buffers: dict[tuple[int, str], dict] = {}
_album_lock = asyncio.Lock()


def _is_authorized(user_id: int | None) -> bool:
    if not ALLOWED_USER_IDS:
        return True  # whitelist disabled
    return user_id in ALLOWED_USER_IDS


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _recent_tail(n: int) -> tuple[str, int]:
    """Return (last n lines of recent.jsonl, total line count). Both empty/0
    if the file is missing."""
    try:
        text = RECENT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "", 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:]), len(lines)


def _build_prompt(
    user_text: str,
    *,
    image_paths: list[str] | None = None,
    unsupported_kind: str | None = None,
) -> str:
    """Inject memory/* directly so claude doesn't burn tool round-trips Reading
    them. Each Read in -p mode is a full inference round-trip — eliminating the
    three (facts/summary/recent) shaves ~15-40s off each turn.

    image_paths: absolute paths to image files attached to this turn. Each gets
    a sibling <image path="..."/> tag and Cas is told to Read them.
    unsupported_kind: human label like '语音'/'视频' when Li sent a media type
    we can't ingest yet — replaces user_message body with an apology stub so
    Cas keeps her own voice instead of a static fallback."""
    facts = _read_text(FACTS_PATH).strip()
    summary = _read_text(SUMMARY_PATH).strip()
    recent_tail, recent_total = _recent_tail(RECENT_TAIL_LINES)

    if unsupported_kind:
        user_block = (
            f"(Li 给你发了一条 {unsupported_kind}，你现在还接收不了这种媒体——"
            "用你自己的语气跟她说一下，不要套话)"
        )
    else:
        user_block = user_text

    image_block = ""
    if image_paths:
        tags = "\n".join(f'<image path="{p}"/>' for p in image_paths)
        image_block = (
            f"\n\n{tags}\n"
            "先用 Read 把上面每个 image 文件读了再决定怎么回。"
            "回复里要让人感觉到你看到了什么——不必刻意'描述图片'，"
            "但不要让未来翻 recent.jsonl 的你自己看到一条只有 [图片]/'嗯' 的空洞 turn。"
        )

    return (
        f"<facts>\n{facts}\n</facts>\n\n"
        f"<summary>\n{summary}\n</summary>\n\n"
        f'<recent total_lines="{recent_total}" showing_last="{RECENT_TAIL_LINES}">\n'
        f"{recent_tail}\n</recent>\n\n"
        "A Telegram user just sent you the message in <user_message>. "
        "Context is already loaded above as <facts>, <summary>, <recent> — "
        "do NOT Read memory/facts.md, memory/summary.md, or "
        "memory/recent.jsonl yourself, that's wasted round-trips. "
        "Follow CLAUDE.md for everything else: decide your reply, Write a "
        "long-term fact to memory/facts.md if warranted, then — as the very "
        "LAST thing — output the reply as plain text. "
        "Do NOT write to memory/recent.jsonl yourself; the orchestrator "
        "records each turn after Telegram confirms delivery. "
        "Compression: total_lines on <recent> is recent.jsonl's full length. "
        "Only if it exceeds 200 should you Read the full file, summarize the "
        "earliest 100 lines into memory/summary.md, and trim recent.jsonl. "
        "Otherwise leave both alone. "
        "Do NOT end on a tool call; the last plain-text message is what "
        "Telegram receives. No preamble, no code fences, no meta-commentary.\n\n"
        f"<user_message>\n{user_block}\n</user_message>"
        f"{image_block}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_turn(
    user_text: str,
    user_ts: str,
    assistant_text: str,
    assistant_ts: str,
    *,
    image_path: str | None = None,
) -> None:
    """Append the user/assistant pair to memory/recent.jsonl. Called only after
    Telegram has accepted the reply, so what's in memory always matches what the
    user saw. The per-chat lock + the fact that claude has already exited means
    no concurrent writer can race us.

    image_path is recorded on the user row when this turn carried a photo. It's
    intentionally NOT surfaced in the <recent> tail injection — Cas only sees the
    [图片] caption placeholder there. The path on disk exists for future cleanup
    hooks (when memory compression moves to cron) and for ad-hoc forensics."""
    user_row = {"role": "user", "content": user_text, "ts": user_ts}
    if image_path:
        user_row["image_path"] = image_path
    lines = [
        json.dumps(user_row, ensure_ascii=False),
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


async def _call_claude(
    user_text: str,
    *,
    image_paths: list[str] | None = None,
    unsupported_kind: str | None = None,
) -> str:
    """Invoke `claude -p` and return its stdout. Raises on timeout / non-zero exit."""
    args = _build_args(
        _build_prompt(user_text, image_paths=image_paths, unsupported_kind=unsupported_kind)
    )
    log.info("spawn claude pid=? cwd=%s model=%s", PROJECT_ROOT, CLAUDE_MODEL or "(default)")
    started = time.monotonic()
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

    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"claude exited {proc.returncode} after {elapsed:.1f}s: "
            f"{err[:500] or '<no stderr>'}"
        )
    log.info("claude completed in %.1fs", elapsed)
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


async def _run_turn(
    msg,
    user_ts: str,
    *,
    prompt_text: str,
    record_text: str,
    image_paths: list[str] | None = None,
    unsupported_kind: str | None = None,
    record_image_path: str | None = None,
) -> None:
    """Single funnel for every kind of incoming message — text / sticker / photo
    / unsupported. Acquires the per-chat lock, drives the typing indicator,
    invokes claude, sends the reply (with retry), and records the turn.

    prompt_text:   what goes inside <user_message> in the prompt to Cas.
    record_text:   what goes into recent.jsonl as user.content. Usually equal
                   to prompt_text; differs for unsupported media where the
                   prompt asks Cas to acknowledge but recent.jsonl just stores
                   a [语音]/[视频]/... placeholder.
    image_paths:   files for Cas to Read this turn.
    unsupported_kind: media-type label when prompt_text should be replaced
                   with the apology stub inside _build_prompt.
    record_image_path: path stored on the user row of recent.jsonl (for
                   future cleanup hooks). Only the primary photo of an album
                   is recorded."""
    chat_id = msg.chat_id
    bot = msg.get_bot()
    async with _chat_locks[chat_id]:
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(_typing_loop(bot, chat_id, stop_typing))
        try:
            try:
                reply = await _call_claude(
                    prompt_text,
                    image_paths=image_paths,
                    unsupported_kind=unsupported_kind,
                )
            except asyncio.TimeoutError:
                log.error("claude timeout chat=%s", chat_id)
                await msg.reply_text(TIMEOUT_REPLY)
                return
            except Exception:
                log.exception("claude failure chat=%s", chat_id)
                await msg.reply_text(ERROR_REPLY)
                return
        finally:
            # stop_typing.set() alone isn't enough: if typing_task is mid
            # send_chat_action when we get here, awaiting it blocks until that
            # in-flight HTTP request completes — which we've seen drag out to
            # 25-30s when the TG API is slow. Cancel it instead so the awaiting
            # request gets interrupted. Note CancelledError is a BaseException,
            # not Exception, so it must be in the except tuple — otherwise it
            # escapes the finally and the chat lock leaks.
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except (asyncio.CancelledError, Exception):
                pass

        if not reply:
            log.warning("claude returned empty reply chat=%s", chat_id)
            await msg.reply_text(EMPTY_REPLY)
            return

        for chunk in _chunks(reply, TELEGRAM_MSG_LIMIT):
            await _send_with_retry(msg, chunk)
        _append_turn(record_text, user_ts, reply, _now_iso(), image_path=record_image_path)
        log.info("sent chat=%s reply_len=%d", chat_id, len(reply))


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.message
    if user is None or msg is None or msg.text is None:
        return
    if not _is_authorized(user.id):
        log.warning(
            "rejected text user_id=%s username=%s text=%r",
            user.id, user.username, msg.text[:60],
        )
        return
    text = msg.text
    user_ts = _now_iso()
    log.info("recv kind=text chat=%s user=%s len=%d", msg.chat_id, user.id, len(text))
    await _run_turn(msg, user_ts, prompt_text=text, record_text=text)


async def on_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.message
    if user is None or msg is None or msg.sticker is None:
        return
    if not _is_authorized(user.id):
        log.warning("rejected sticker user_id=%s", user.id)
        return
    emoji = msg.sticker.emoji or ""
    text = f"[贴纸] {emoji}".rstrip()
    user_ts = _now_iso()
    log.info("recv kind=sticker chat=%s user=%s emoji=%r", msg.chat_id, user.id, emoji)
    await _run_turn(msg, user_ts, prompt_text=text, record_text=text)


_UNSUPPORTED_LABELS = [
    ("voice", "语音"),
    ("video", "视频"),
    ("audio", "音频"),
    ("video_note", "视频"),
    ("document", "文件"),
]


def _unsupported_kind(msg) -> str | None:
    for attr, label in _UNSUPPORTED_LABELS:
        if getattr(msg, attr, None):
            return label
    return None


async def on_unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.message
    if user is None or msg is None:
        return
    if not _is_authorized(user.id):
        log.warning("rejected unsupported media user_id=%s", user.id)
        return
    kind = _unsupported_kind(msg)
    if kind is None:
        return  # filter matched but we can't classify — let it drop silently
    user_ts = _now_iso()
    log.info("recv kind=unsupported(%s) chat=%s user=%s", kind, msg.chat_id, user.id)
    # prompt_text is unused when unsupported_kind is set (build_prompt swaps it),
    # but keep it informative for log-level grep.
    await _run_turn(
        msg,
        user_ts,
        prompt_text=f"[{kind}]",
        record_text=f"[{kind}]",
        unsupported_kind=kind,
    )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.message
    if user is None or msg is None or not msg.photo:
        return
    if not _is_authorized(user.id):
        log.warning("rejected photo user_id=%s", user.id)
        return

    chat_id = msg.chat_id
    photo = msg.photo[-1]  # largest size TG generates
    path = MEDIA_DIR / f"{chat_id}_{msg.message_id}.jpg"
    caption = msg.caption or ""
    log.info(
        "recv kind=photo chat=%s msg=%s media_group=%s caption_len=%d",
        chat_id, msg.message_id, msg.media_group_id, len(caption),
    )

    try:
        tg_file = await context.bot.get_file(photo.file_id)
        await tg_file.download_to_drive(custom_path=str(path))
    except Exception:
        log.exception("photo download failed chat=%s msg=%s", chat_id, msg.message_id)
        try:
            await msg.reply_text(ERROR_REPLY)
        except Exception:
            log.exception("error reply also failed chat=%s", chat_id)
        return

    user_ts = _now_iso()

    if not msg.media_group_id:
        # Single photo: process now.
        record_text = f"[图片] {caption}".rstrip()
        await _run_turn(
            msg,
            user_ts,
            prompt_text=record_text,
            record_text=record_text,
            image_paths=[str(path)],
            record_image_path=str(path),
        )
        return

    # Album: buffer + debounce. First photo of a group spawns the flush task;
    # subsequent photos within ALBUM_DEBOUNCE_S extend the deadline.
    key = (chat_id, msg.media_group_id)
    spawn_flush = False
    async with _album_lock:
        buf = _album_buffers.get(key)
        if buf is None:
            buf = {
                "msg": msg,
                "paths": [str(path)],
                "caption": caption,
                "user_ts": user_ts,
                "last_seen": time.monotonic(),
            }
            _album_buffers[key] = buf
            spawn_flush = True
        else:
            buf["paths"].append(str(path))
            if caption and not buf["caption"]:
                # Captions on TG albums usually ride only one of the photos.
                buf["caption"] = caption
            buf["last_seen"] = time.monotonic()
    if spawn_flush:
        asyncio.create_task(_flush_album_when_quiet(key))


async def _flush_album_when_quiet(key: tuple[int, str]) -> None:
    """Sleep until ALBUM_DEBOUNCE_S has passed since the last photo in this
    group, then run a single Cas turn over all of them."""
    while True:
        async with _album_lock:
            buf = _album_buffers.get(key)
            if buf is None:
                return
            wait = buf["last_seen"] + ALBUM_DEBOUNCE_S - time.monotonic()
        if wait <= 0:
            break
        await asyncio.sleep(wait)
    async with _album_lock:
        buf = _album_buffers.pop(key, None)
    if buf is None:
        return
    paths = buf["paths"]
    caption = buf["caption"]
    msg = buf["msg"]
    record_text = f"[图片]{f' {caption}' if caption else ''}"
    log.info("album flush chat=%s group=%s n_photos=%d", key[0], key[1], len(paths))
    await _run_turn(
        msg,
        buf["user_ts"],
        prompt_text=record_text,
        record_text=record_text,
        image_paths=paths,
        record_image_path=paths[0],
    )


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
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Sticker.ALL, on_sticker))
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.VIDEO | filters.AUDIO | filters.VIDEO_NOTE | filters.Document.ALL,
            on_unsupported,
        )
    )
    app.add_error_handler(on_error)
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
