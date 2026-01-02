import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import aiosqlite
import dateparser
import joblib
import yake
from lxml import html
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm


# ============================================================
# НАСТРОЙКИ (шапка)
# ============================================================
# Диапазон. Если пусто => парсить 1..latest
start_search = ""   # например "1573896"
stop_search = ""    # например "1573996" (включительно)

# Параллельность/тайминги
CONCURRENCY = 6
REQUEST_TIMEOUT_SEC = 25
SLEEP_BETWEEN_REQUESTS_SEC = 0.35

# Политика капчи: "stop" или "skip"
CAPTCHA_POLICY = "stop"

# Минимальная длина текста статьи (иначе SKIP)
MIN_TEXT_LEN = 200

# Сколько тегов извлекать
TAGS_TOPK = 10

# Длина краткого описания
SUMMARY_CHARS = 420

# Частота коммитов/сохранения модели
COMMIT_EACH = 50
SAVE_MODEL_EACH = 200

# Порог уверенности эвристики, чтобы self-train
HEUR_TRAIN_THRESHOLD = 2

# Лимит размера одного sqlite файла
MAX_DB_BYTES = 35 * 1024 * 1024

# Логи
LOG_LEVEL = "INFO"   # DEBUG/INFO/WARNING/ERROR
LOG_FILE = "aftershock_parser.log"

# Юзер-агент
USER_AGENT = "Mozilla/5.0 (compatible; research-bot; +https://example.com)"
# ============================================================


BASE = "https://aftershock.news/"
NODE_URL = "https://aftershock.news/?q=node/{nid}&full"
PULSE_URL = "https://aftershock.news/?q=all"

MODEL_PATH = "model_direction.joblib"

LABELS = ["economy", "politics", "military", "other"]

CAPTCHA_MARKERS = [
    "captcha", "recaptcha", "cloudflare", "verify you are",
    "докажите, что вы не робот", "проверка браузера",
]

ECON_WORDS = {
    "инфляц", "ставк", "ввп", "рубл", "доллар", "евро", "рынок", "акци",
    "облигац", "санкц", "нефть", "газ", "бюджет", "налог", "банк", "цб",
    "курс", "экспорт", "импорт", "дефицит", "профицит", "кризис", "пошлин",
}
MIL_WORDS = {
    "сво", "фронт", "всу", "арм", "ракет", "дрон", "артилл", "танк",
    "пво", "авиац", "штурм", "наступ", "удар", "обстрел", "мобилиз",
    "боеприпас", "пленный",
}
POL_WORDS = {
    "выбор", "президент", "парламент", "парт", "переговор",
    "дипломат", "закон", "правительств", "министр", "оппозиц", "протест",
    "нато", "оон", "сша", "ес", "китай", "иран", "саммит",
}


@dataclass
class Article:
    nid: int
    url: str
    title: str | None
    published_at: str | None
    published_raw: str | None
    text: str
    summary: str | None
    tags_json: str
    direction: str
    direction_confidence: float | None
    sha1: str


# ----------------------------
# Utils
# ----------------------------
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
    )


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def parse_published_datetime(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    raw2 = clean_text(raw)
    dt = dateparser.parse(
        raw2,
        languages=["ru"],
        settings={"DATE_ORDER": "DMY", "TIMEZONE": "Europe/Moscow", "RETURN_AS_TIMEZONE_AWARE": True},
    )
    if not dt:
        return None, raw2
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(), raw2


def extract_best_date(tree: html.HtmlElement) -> tuple[str | None, str | None]:
    meta = tree.xpath("//meta[@property='article:published_time']/@content")
    if meta:
        iso, raw = parse_published_datetime(meta[0])
        if iso:
            return iso, raw

    spans = tree.xpath("//span[contains(@class,'aft-postdateicon')]//text()")
    if spans:
        raw = " ".join(clean_text(x) for x in spans if clean_text(x))
        iso, raw = parse_published_datetime(raw)
        if iso:
            return iso, raw

    return None, None


def extract_title(tree: html.HtmlElement) -> str | None:
    t = tree.xpath("//article//h1[contains(@class,'aft-postheader')]//text()")
    t = [clean_text(x) for x in t if clean_text(x)]
    return t[0] if t else None


def extract_body_text(tree: html.HtmlElement) -> str:
    body = tree.xpath("//div[contains(@class,'field-name-body')]")
    if not body:
        return ""
    return clean_text(body[0].text_content())


def keyword_tags_ru(text: str, top_k: int = 10) -> list[str]:
    kw = yake.KeywordExtractor(lan="ru", n=2, top=top_k)
    out = []
    for phrase, score in kw.extract_keywords(text[:20000]):
        p = clean_text(phrase.lower())
        if len(p) >= 3:
            out.append(p)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:top_k]


def heuristic_direction(text: str) -> tuple[str, int]:
    t = text.lower()
    score_e = sum(1 for w in ECON_WORDS if w in t)
    score_m = sum(1 for w in MIL_WORDS if w in t)
    score_p = sum(1 for w in POL_WORDS if w in t)

    best_score, best_label = max(
        [(score_e, "economy"), (score_p, "politics"), (score_m, "military")],
        key=lambda x: x[0],
    )
    if best_score >= 2:
        return best_label, best_score
    return "other", best_score


def simple_summary(text: str, max_chars: int = 420) -> str:
    t = clean_text(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars].strip()


def is_soft_404(html_txt: str) -> bool:
    low = html_txt.lower()
    if "страница не найдена" in low:
        return True
    if "запрашиваемая страница" in low and "не найдена" in low:
        return True

    try:
        tree = html.fromstring(html_txt)
        h1 = tree.xpath("//div[contains(@class,'aft-postcontent')]//h1/text()")
        h1 = [clean_text(x).lower() for x in h1 if clean_text(x)]
        if any(x == "страница не найдена" for x in h1):
            return True
    except Exception:
        pass

    return False


# ----------------------------
# Model
# ----------------------------
def load_or_init_model():
    vect = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2")
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj["vect"], obj["clf"]

    clf = SGDClassifier(loss="log_loss", alpha=1e-6, random_state=42)
    X0 = vect.transform(["init"])
    clf.partial_fit(X0, ["other"], classes=LABELS)
    return vect, clf


def save_model(vect, clf):
    joblib.dump({"vect": vect, "clf": clf}, MODEL_PATH)


# ----------------------------
# DB (with migrations)
# ----------------------------
def db_name(part: int) -> str:
    return f"aftershock_articles_{part:03d}.sqlite"


async def _column_exists(db: aiosqlite.Connection, table: str, column: str) -> bool:
    cur = await db.execute(f"PRAGMA table_info({table})")
    rows = await cur.fetchall()
    await cur.close()
    return any(r[1] == column for r in rows)  # r[1] is column name


async def _add_column_if_missing(db: aiosqlite.Connection, table: str, column: str, decl: str):
    if not await _column_exists(db, table, column):
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        await db.commit()


async def ensure_schema(db: aiosqlite.Connection):
    await db.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS articles (
            nid INTEGER PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            published_at TEXT,
            published_raw TEXT,
            text TEXT NOT NULL,
            summary TEXT,
            tags_json TEXT NOT NULL,
            direction TEXT NOT NULL,
            direction_confidence REAL,
            sha1 TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
        CREATE INDEX IF NOT EXISTS idx_articles_direction ON articles(direction);
        """
    )
    await db.commit()

    # migrations for older db files
    await _add_column_if_missing(db, "articles", "direction_confidence", "REAL")


async def open_db(part: int) -> aiosqlite.Connection:
    path = db_name(part)
    db = await aiosqlite.connect(path)
    await ensure_schema(db)
    return db


async def save_article(db: aiosqlite.Connection, a: Article):
    await db.execute(
        """
        INSERT OR REPLACE INTO articles
        (nid, url, title, published_at, published_raw, text, summary, tags_json, direction, direction_confidence, sha1, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            a.nid,
            a.url,
            a.title,
            a.published_at,
            a.published_raw,
            a.text,
            a.summary,
            a.tags_json,
            a.direction,
            a.direction_confidence,
            a.sha1,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


# ----------------------------
# Networking / parsing
# ----------------------------
async def get_latest_nid(session: aiohttp.ClientSession) -> int | None:
    try:
        async with session.get(PULSE_URL, timeout=REQUEST_TIMEOUT_SEC) as r:
            if r.status != 200:
                return None
            html_txt = await r.text(errors="ignore")
    except Exception:
        return None

    m = re.findall(r"\?q=node/(\d+)", html_txt)
    if not m:
        return None
    return max(int(x) for x in m)


async def fetch_html(session: aiohttp.ClientSession, nid: int, max_retries: int = 5):
    """
    Returns: (status, html_text_or_None, reason)
    """
    url = NODE_URL.format(nid=nid)

    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=REQUEST_TIMEOUT_SEC) as r:
                status = r.status
                txt = await r.text(errors="ignore")

                if status == 404:
                    return status, None, "404"

                if status in (429, 403, 503):
                    ra = r.headers.get("Retry-After")
                    wait = int(ra) if (ra and ra.isdigit()) else (2 ** attempt)
                    logging.warning(f"RETRY nid={nid} status={status} wait={wait}s attempt={attempt+1}/{max_retries}")
                    await asyncio.sleep(wait)
                    continue

                if status != 200:
                    return status, None, f"http_{status}"

                low = txt.lower()
                if any(m in low for m in CAPTCHA_MARKERS):
                    return status, None, "captcha_detected"

                if is_soft_404(txt):
                    return status, None, "soft_404_page"

                await asyncio.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
                return status, txt, "ok"

        except asyncio.TimeoutError:
            wait = 2 ** attempt
            logging.warning(f"RETRY nid={nid} reason=timeout wait={wait}s attempt={attempt+1}/{max_retries}")
            await asyncio.sleep(wait)
            continue
        except aiohttp.ClientError as e:
            wait = 2 ** attempt
            logging.warning(f"RETRY nid={nid} reason=client_error={type(e).__name__} wait={wait}s attempt={attempt+1}/{max_retries}")
            await asyncio.sleep(wait)
            continue
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"RETRY nid={nid} reason=error={type(e).__name__} wait={wait}s attempt={attempt+1}/{max_retries}")
            await asyncio.sleep(wait)
            continue

    return None, None, "max_retries_exceeded"


def parse_article(nid: int, url: str, html_txt: str):
    tree = html.fromstring(html_txt)
    title = extract_title(tree)
    published_at, published_raw = extract_best_date(tree)
    text = extract_body_text(tree)
    return title, published_at, published_raw, text


# ----------------------------
# Worker pipeline (queue-based, scalable)
# ----------------------------
async def persist_article(state: dict, article: Article):
    async with state["db_lock"]:
        db_path = db_name(state["db_part"])
        if os.path.exists(db_path) and os.path.getsize(db_path) > MAX_DB_BYTES:
            await state["db"].commit()
            await state["db"].close()
            state["db_part"] += 1
            state["db"] = await open_db(state["db_part"])
            logging.warning(f"DB rotated -> {db_name(state['db_part'])}")

        await save_article(state["db"], article)
        state["saved"] += 1

        if state["saved"] % COMMIT_EACH == 0:
            await state["db"].commit()

        if state["saved"] % SAVE_MODEL_EACH == 0:
            save_model(state["vect"], state["clf"])
            logging.info(f"MODEL saved: saved={state['saved']} trained={state['trained']}")


async def worker(name: str, q: asyncio.Queue, session: aiohttp.ClientSession, state: dict, pbar: tqdm, pbar_lock: asyncio.Lock):
    while True:
        nid = await q.get()
        if nid is None:
            q.task_done()
            return

        try:
            url = NODE_URL.format(nid=nid)
            status, html_txt, reason = await fetch_html(session, nid)
            state["processed"] += 1

            if reason == "captcha_detected":
                state["captcha"] += 1
                logging.error(f"CAPTCHA nid={nid} url={url}")
                if CAPTCHA_POLICY == "stop":
                    raise RuntimeError("CAPTCHA detected. Stopping by policy.")
                state["skipped"] += 1
                continue

            if not html_txt:
                state["skipped"] += 1
                state["skip_reasons"][reason] = state["skip_reasons"].get(reason, 0) + 1
                logging.info(f"SKIP nid={nid} status={status} reason={reason}")
                continue

            try:
                title, published_at, published_raw, text = parse_article(nid, url, html_txt)
            except Exception as e:
                state["skipped"] += 1
                state["skip_reasons"]["parse_error"] = state["skip_reasons"].get("parse_error", 0) + 1
                logging.info(f"SKIP nid={nid} reason=parse_error err={type(e).__name__}")
                continue

            text = clean_text(text)
            if not title or len(text) < MIN_TEXT_LEN:
                state["skipped"] += 1
                state["skip_reasons"]["parsed_bad"] = state["skip_reasons"].get("parsed_bad", 0) + 1
                logging.info(f"SKIP nid={nid} reason=parsed_bad title={bool(title)} text_len={len(text)}")
                continue

            tags = keyword_tags_ru(text, top_k=TAGS_TOPK)
            heur_label, heur_score = heuristic_direction(text)

            X = state["vect"].transform([text])
            direction = heur_label
            conf = None
            try:
                proba = state["clf"].predict_proba(X)[0]
                pred = LABELS[int(proba.argmax())]
                conf = float(proba.max())
                if heur_score < HEUR_TRAIN_THRESHOLD:
                    direction = pred
            except Exception:
                conf = None

            if heur_score >= HEUR_TRAIN_THRESHOLD:
                try:
                    state["clf"].partial_fit(X, [heur_label])
                    state["trained"] += 1
                except Exception:
                    pass

            summary = simple_summary(text, max_chars=SUMMARY_CHARS)
            sha1 = sha1_text((title or "") + "\n" + text[:4000])

            article = Article(
                nid=nid,
                url=url,
                title=title,
                published_at=published_at,
                published_raw=published_raw,
                text=text,
                summary=summary,
                tags_json=json.dumps(["#" + t.replace(" ", "_") for t in tags], ensure_ascii=False),
                direction=direction,
                direction_confidence=conf,
                sha1=sha1,
            )

            await persist_article(state, article)

            if state["saved"] % 25 == 0:
                logging.info(
                    f"OK nid={nid} saved={state['saved']} processed={state['processed']} skipped={state['skipped']} "
                    f"dir={direction} conf={conf}"
                )

        finally:
            q.task_done()
            async with pbar_lock:
                pbar.update(1)
                pbar.set_postfix(
                    saved=state["saved"],
                    skipped=state["skipped"],
                    captcha=state["captcha"],
                    trained=state["trained"],
                )


async def main():
    setup_logging()

    vect, clf = load_or_init_model()

    state = {
        "vect": vect,
        "clf": clf,

        "db": None,
        "db_part": 1,
        "db_lock": asyncio.Lock(),

        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "captcha": 0,
        "trained": 0,
        "skip_reasons": {},
    }

    state["db"] = await open_db(state["db_part"])

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
    headers = {"User-Agent": USER_AGENT}

    q: asyncio.Queue[int | None] = asyncio.Queue(maxsize=CONCURRENCY * 10)
    pbar_lock = asyncio.Lock()

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        latest = await get_latest_nid(session)
        if not latest:
            raise RuntimeError("Не удалось определить latest_nid через /?q=all")

        if start_search.strip() and stop_search.strip():
            start = int(start_search)
            end = int(stop_search)
        elif start_search.strip() and not stop_search.strip():
            start = int(start_search)
            end = latest
        elif not start_search.strip() and stop_search.strip():
            start = 1
            end = int(stop_search)
        else:
            start = 1
            end = latest

        if start > end:
            raise ValueError(f"start_search({start}) > stop_search({end})")

        total = end - start + 1
        logging.info(f"Latest nid ≈ {latest}. Run range: {start}..{end} (≈{total} шт.)")
        logging.info(
            f"Concurrency={CONCURRENCY} timeout={REQUEST_TIMEOUT_SEC}s sleep={SLEEP_BETWEEN_REQUESTS_SEC}s captcha_policy={CAPTCHA_POLICY}"
        )

        pbar = tqdm(total=total, desc="Processed", unit="node")

        workers = [
            asyncio.create_task(worker(f"w{i+1}", q, session, state, pbar, pbar_lock))
            for i in range(CONCURRENCY)
        ]

        try:
            for nid in range(start, end + 1):
                await q.put(nid)

            for _ in workers:
                await q.put(None)

            await q.join()

        except RuntimeError as e:
            logging.error(str(e))
            raise

        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

            async with pbar_lock:
                pbar.close()

    await state["db"].commit()
    await state["db"].close()
    save_model(state["vect"], state["clf"])

    logging.info(
        f"Done. processed={state['processed']} saved={state['saved']} skipped={state['skipped']} "
        f"captcha={state['captcha']} trained={state['trained']} skip_reasons={state['skip_reasons']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
