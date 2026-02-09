import os
import io
import re
import json
import asyncio
import string
from typing import Any, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from openai import OpenAI
from pydantic import BaseModel, Field


# ====================== CONFIG ======================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "gpt-4.1-mini").strip()

MAX_CODES_PER_ANSWER = int(os.getenv("MAX_CODES_PER_ANSWER", "3"))
BATCH_SIZE_FOR_CODEBOOK = int(os.getenv("BATCH_SIZE_FOR_CODEBOOK", "300"))
BATCH_SIZE_FOR_CODING = int(os.getenv("BATCH_SIZE_FOR_CODING", "50"))

REQUIRED_CODES = [
    ("Затрудняюсь ответить", "Респондент пишет, что не знает/не может ответить/затрудняется."),
    ("Другое/не подходит", "Ответ не подходит ни под один код / слишком общий / вне темы."),
]

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# ====================== SCHEMAS ======================

class CodebookItem(BaseModel):
    code: str = Field(..., description="Short code name in Russian")
    description: str = Field(..., description="Brief include/exclude definition")

class CodebookResponse(BaseModel):
    codes: List[CodebookItem]

class CodingRow(BaseModel):
    codes: List[str]
    comment: str

class CodingBatchResponse(BaseModel):
    rows: List[CodingRow]


# ====================== FSM ======================

class Flow(StatesGroup):
    waiting_question = State()
    waiting_file = State()
    waiting_column_letter = State()
    waiting_codes_choice = State()
    waiting_codes_manual = State()
    waiting_max_codes = State()
    reviewing_codebook = State()
    coding = State()


# ====================== HELPERS ======================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]

def parse_manual_codebook(text: str) -> List[CodebookItem]:
    """
    Lines:
      Код — описание
      Код - описание
      Код: описание
    """
    items: List[CodebookItem] = []
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for line in lines:
        parts = re.split(r"\s*[—-:]\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        code, desc = parts[0].strip(), parts[1].strip()
        if code and desc:
            items.append(CodebookItem(code=code, description=desc))
    return items

def ensure_required_codes(codebook: List[CodebookItem]) -> List[CodebookItem]:
    existing = {c.code.strip().lower() for c in codebook}
    out = list(codebook)
    for code, desc in REQUIRED_CODES:
        if code.lower() not in existing:
            out.append(CodebookItem(code=code, description=desc))
    # dedup by code lower
    seen = set()
    dedup: List[CodebookItem] = []
    for c in out:
        key = c.code.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup

def format_codebook(codebook: List[CodebookItem]) -> str:
    return "\n".join([f"{i}. {c.code} — {c.description}" for i, c in enumerate(codebook, 1)])

def make_codes_choice_keyboard():
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Коды уже есть", callback_data="codes::have")
    kb.button(text="✨ Сгенерируй коды", callback_data="codes::gen")
    kb.adjust(1)
    return kb.as_markup()

def make_max_codes_keyboard():
    kb = InlineKeyboardBuilder()
    for n in [8, 10, 12, 15, 20, 25]:
        kb.button(text=str(n), callback_data=f"max::{n}")
    kb.adjust(3)
    return kb.as_markup()

def make_review_keyboard(can_regen: bool):
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Ок, кодируем", callback_data="review::ok")
    if can_regen:
        kb.button(text="🔁 Перегенерить коды", callback_data="review::regen")
    kb.button(text="🧹 Сброс (reset)", callback_data="review::reset")
    kb.adjust(1)
    return kb.as_markup()

def apply_edit_command(codebook: List[CodebookItem], cmd: str) -> Tuple[List[CodebookItem], str]:
    """
    Commands:
      help
      add <код> — <описание>
      rename <номер> <новое_имя>
      desc <номер> <новое_описание>
      del <номер>
    """
    s = (cmd or "").strip()
    if not s:
        return codebook, "Пустая команда."

    low = s.lower()

    if low == "help":
        return codebook, (
            "Команды редактирования:\n"
            "• add <код> — <описание>\n"
            "• rename <номер> <новое_имя>\n"
            "• desc <номер> <новое_описание>\n"
            "• del <номер>\n"
            "• help\n\n"
            "Пример: add Цена — Про дороговизну/выгоду"
        )

    if low.startswith("add "):
        rest = s[4:].strip()
        items = parse_manual_codebook(rest)
        if not items:
            return codebook, "Не понял формат. Пример: add Цена — Дорого/выгодно"
        return ensure_required_codes(codebook + items), "Добавил(а)."

    if low.startswith("rename "):
        m = re.match(r"rename\s+(\d+)\s+(.+)$", s, flags=re.IGNORECASE)
        if not m:
            return codebook, "Формат: rename <номер> <новое_имя>"
        idx = int(m.group(1)) - 1
        name = m.group(2).strip()
        if idx < 0 or idx >= len(codebook):
            return codebook, "Неверный номер."
        codebook[idx].code = name
        return ensure_required_codes(codebook), "Переименовал(а)."

    if low.startswith("desc "):
        m = re.match(r"desc\s+(\d+)\s+(.+)$", s, flags=re.IGNORECASE)
        if not m:
            return codebook, "Формат: desc <номер> <новое_описание>"
        idx = int(m.group(1)) - 1
        desc = m.group(2).strip()
        if idx < 0 or idx >= len(codebook):
            return codebook, "Неверный номер."
        codebook[idx].description = desc
        return codebook, "Описание обновлено."

    if low.startswith("del ") or low.startswith("delete "):
        m = re.match(r"(del|delete)\s+(\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return codebook, "Формат: del <номер>"
        idx = int(m.group(2)) - 1
        if idx < 0 or idx >= len(codebook):
            return codebook, "Неверный номер."
        removed = codebook[idx].code
        new_cb = [c for i, c in enumerate(codebook) if i != idx]
        return ensure_required_codes(new_cb), f"Удалил(а) «{removed}»."

    return codebook, "Не понял команду. Напиши `help`."

def columns_letter_map(cols: List[str]) -> List[Tuple[str, str]]:
    letters = list(string.ascii_uppercase)
    pairs: List[Tuple[str, str]] = []
    for i, col in enumerate(cols):
        if i < len(letters):
            pairs.append((letters[i], col))
        else:
            a = letters[(i // 26) - 1]
            b = letters[i % 26]
            pairs.append((a + b, col))
    return pairs

def render_columns_menu(pairs: List[Tuple[str, str]]) -> str:
    lines = ["Файл получен ✅", "Теперь введи букву столбца с ответами (например, A):", ""]
    for letter, col in pairs[:80]:
        lines.append(f"{letter}) {col}")
    if len(pairs) > 80:
        lines.append("… (показаны не все колонки)")
    return "\n".join(lines)

def chat_json(model: str, messages: list, json_schema: dict) -> dict:
    """
    Chat Completions + JSON schema (works with openai==1.61.0).
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {"name": "schema", "schema": json_schema}},
    )
    content = resp.choices[0].message.content or ""
    return json.loads(content)


# ====================== OPENAI CALLS ======================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def llm_topics_for_chunk(question: str, answers_chunk: List[str]) -> str:
    content = "\n".join([f"- {t}" for t in answers_chunk if t])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ты исследователь. Выделяй устойчивые темы из открытых ответов."},
            {"role": "user", "content": (
                f"Контекст вопроса анкеты:\n{question}\n\n"
                f"Ответы пользователей:\n{content}\n\n"
                "Верни список из 10-20 коротких тем (фразами), без пояснений, по одной теме в строке."
            )},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def llm_make_codebook(question: str, all_texts: List[str], max_codes: int) -> List[CodebookItem]:
    # sample up to 2000 answers
    if len(all_texts) > 2000:
        step = max(1, len(all_texts) // 2000)
        sample = all_texts[::step][:2000]
    else:
        sample = all_texts

    chunks = chunk_list(sample, BATCH_SIZE_FOR_CODEBOOK)
    topic_lists: List[str] = []
    for ch in chunks:
        topic_lists.append(llm_topics_for_chunk(question, ch))

    merged_topics = "\n".join(topic_lists)

    schema = CodebookResponse.model_json_schema()
    data = chat_json(
        MODEL,
        messages=[
            {"role": "system", "content": (
                "Ты создаешь кодбук для кодировки открытых ответов. "
                "Коды короткие. Взаимоисключаемость не обязательна (multi-code)."
            )},
            {"role": "user", "content": (
                f"Контекст вопроса анкеты:\n{question}\n\n"
                f"Список тем из ответов (сырой):\n{merged_topics}\n\n"
                f"Сформируй финальный список кодов (максимум {max_codes}). "
                "У каждого кода: короткое название и описание (что включаем/что не включаем в 1-2 фразах). "
                "Обязательно добавь коды 'Затрудняюсь ответить' и 'Другое/не подходит'. "
                "Верни строго JSON по схеме."
            )},
        ],
        json_schema=schema
    )
    parsed = CodebookResponse.model_validate(data)
    codebook = ensure_required_codes(parsed.codes)

    # trim but keep required
    if len(codebook) > max_codes:
        required_names = {c[0].lower() for c in REQUIRED_CODES}
        required_items = [c for c in codebook if c.code.lower() in required_names]
        other_items = [c for c in codebook if c.code.lower() not in required_names]
        keep_n = max(0, max_codes - len(required_items))
        codebook = other_items[:keep_n] + required_items

    return codebook

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def llm_code_batch(question: str, texts: List[str], codebook: List[CodebookItem]) -> List[CodingRow]:
    allowed = [c.code for c in codebook]

    joined = "\n".join([f"{i+1}) {t}" for i, t in enumerate(texts)])
    prompt = (
        f"Контекст вопроса анкеты:\n{question}\n\n"
        "Кодируй ответы multi-code.\n"
        "Правила:\n"
        "- Можно присваивать несколько кодов.\n"
        f"- Максимум {MAX_CODES_PER_ANSWER} кода на один ответ.\n"
        "- Используй ТОЛЬКО коды из списка.\n"
        "- Если пользователь явно пишет 'не знаю/затрудняюсь' — ставь 'Затрудняюсь ответить'.\n"
        "- Если ничего не подходит — 'Другое/не подходит'.\n"
        "- Комментарий: 1 короткое предложение почему.\n\n"
        "Кодбук:\n" + "\n".join([f"- {c.code}: {c.description}" for c in codebook]) +
        f"\n\nОтветы:\n{joined}\n\nВерни JSON."
    )

    schema = CodingBatchResponse.model_json_schema()
    data = chat_json(
        MODEL,
        messages=[
            {"role": "system", "content": "Ты аккуратно кодируешь открытые ответы по заданному кодбуку."},
            {"role": "user", "content": prompt},
        ],
        json_schema=schema
    )

    parsed = CodingBatchResponse.model_validate(data)
    out: List[CodingRow] = []

    for r in parsed.rows[:len(texts)]:
        codes = r.codes[:MAX_CODES_PER_ANSWER] if r.codes else ["Другое/не подходит"]
        codes = [c for c in codes if c in allowed] or ["Другое/не подходит"]
        out.append(CodingRow(codes=codes, comment=(r.comment or "").strip()))

    while len(out) < len(texts):
        out.append(CodingRow(codes=["Другое/не подходит"], comment="Пустой ответ от модели"))

    return out[:len(texts)]


# ====================== BOT ======================

bot = Bot(BOT_TOKEN)
dp = Dispatcher()


# ---------- commands ----------

@dp.message(Command("reset"))
async def cmd_reset(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.waiting_question)
    await msg.answer("Процесс перезапущен. Все предыдущие данные были очищены.\n\nВведите вопрос анкеты (что спрашивали у респондентов).")

@dp.message(Command("cancel"))
async def cmd_cancel(msg: Message, state: FSMContext):
    data = await state.get_data()
    if data.get("cancel_requested"):
        await msg.answer("Отмена уже запрошена. Останавливаю…")
        return
    await state.update_data(cancel_requested=True)
    await msg.answer("Ок, отменяю текущую операцию…")

@dp.message(Command("status"))
async def cmd_status(msg: Message, state: FSMContext):
    data = await state.get_data()
    stage = data.get("progress_stage")
    done = data.get("progress_done")
    total = data.get("progress_total")
    if not stage or done is None or total is None:
        await msg.answer("Сейчас ничего не выполняю. Напиши /start чтобы начать.")
    else:
        await msg.answer(f"Статус: {stage}\nПрогресс: {done}/{total}")


# ---------- start ----------

@dp.message(CommandStart())
async def start(msg: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.waiting_question)
    await msg.answer(
        "Процесс перезапущен. Все предыдущие данные были очищены.\n\n"
        "Введите, пожалуйста, вопрос анкеты (что спрашивали у респондентов).\n"
        "Это нужно, чтобы корректнее генерировать коды и кодировать ответы."
    )

@dp.message(Flow.waiting_question)
async def on_question(msg: Message, state: FSMContext):
    q = clean_text(msg.text or "")
    if len(q) < 10:
        await msg.answer("Вопрос слишком короткий. Пришли, пожалуйста, полный текст вопроса.")
        return
    await state.update_data(question=q, cancel_requested=False)
    await state.set_state(Flow.waiting_file)
    await msg.answer(
        "Вопрос сохранен ✅\n\n"
        "Теперь загрузите файл с данными (.xlsx).\n"
        "В файле должна быть одна вкладка."
    )

# ---------- file ----------

@dp.message(Flow.waiting_file, F.document)
async def on_file(msg: Message, state: FSMContext):
    doc = msg.document
    if not doc.file_name.lower().endswith(".xlsx"):
        await msg.answer("Пока поддерживаю только .xlsx. Пришли, пожалуйста, Excel.")
        return

    file = await bot.get_file(doc.file_id)
    content = await bot.download_file(file.file_path)
    b = content.read()

    try:
        df = pd.read_excel(io.BytesIO(b))
    except Exception as e:
        await msg.answer(f"Не смог прочитать файл: {e}")
        return

    if df.empty:
        await msg.answer("Файл пустой 😕")
        return

    cols = [str(c) for c in df.columns.tolist()]
    pairs = columns_letter_map(cols)

    await state.update_data(
        file_bytes=b,
        df_json=df.to_json(orient="records", force_ascii=False),
        columns_pairs=pairs,
        cancel_requested=False,
        progress_stage=None,
        progress_done=None,
        progress_total=None,
        codebook_generated=False,
    )
    await state.set_state(Flow.waiting_column_letter)

    data = await state.get_data()
    await msg.answer(f"Контекст вопроса:\n{data['question']}\n\n" + render_columns_menu(pairs))

@dp.message(Flow.waiting_column_letter)
async def on_column_letter(msg: Message, state: FSMContext):
    s = (msg.text or "").strip().upper()
    data = await state.get_data()
    pairs = data.get("columns_pairs") or []
    mapping = {k.upper(): v for k, v in pairs}

    if s not in mapping:
        await msg.answer("Не понял букву. Пришли одну букву (A, B, C…) из списка выше.")
        return

    col = mapping[s]
    df = pd.DataFrame(json.loads(data["df_json"]))
    if col not in df.columns:
        await msg.answer("Колонка не найдена. Попробуй /reset и загрузить файл заново.")
        return

    await state.update_data(text_col=col, column_letter=s)
    await state.set_state(Flow.waiting_codes_choice)

    await msg.answer(
        f"Вы выбрали столбец {s} ({col}).\n\nУ вас есть готовые категории для анализа?",
        reply_markup=make_codes_choice_keyboard()
    )

# ---------- codes choice ----------

@dp.callback_query(Flow.waiting_codes_choice, F.data == "codes::have")
async def codes_have(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_codes_manual)
    await cb.message.answer(
        "Ок. Пришлите категории одним сообщением в формате:\n"
        "Код — описание\n"
        "или\n"
        "Код: описание\n\n"
        "Пример:\n"
        "Цена — Про дороговизну/выгоду\n"
        "Удобство — Скорость/простота/понятность\n\n"
        "Важно: я всегда добавлю 'Затрудняюсь ответить' и 'Другое/не подходит', если их нет."
    )
    await cb.answer()

@dp.callback_query(Flow.waiting_codes_choice, F.data == "codes::gen")
async def codes_gen(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.waiting_max_codes)
    await state.update_data(codebook_generated=True)
    await cb.message.answer("Ок. Сколько категорий максимум сгенерировать?", reply_markup=make_max_codes_keyboard())
    await cb.answer()

# ---------- manual codes ----------

@dp.message(Flow.waiting_codes_manual)
async def on_manual_codebook(msg: Message, state: FSMContext):
    items = parse_manual_codebook(msg.text or "")
    if len(items) < 2:
        await msg.answer("Не смог распознать категории. Проверь формат. Можно написать `help`.")
        return

    codebook = ensure_required_codes(items)
    await state.update_data(codebook_json=json.dumps([c.model_dump() for c in codebook], ensure_ascii=False))
    await state.set_state(Flow.reviewing_codebook)

    await msg.answer(
        "Категории сохранены ✅\n\n"
        "Если нужно — отредактируй командами (help), либо нажми ✅ Ок.\n\n"
        + format_codebook(codebook),
        reply_markup=make_review_keyboard(can_regen=False)
    )

# ---------- generated codes ----------

@dp.callback_query(Flow.waiting_max_codes, F.data.startswith("max::"))
async def pick_max_codes(cb: CallbackQuery, state: FSMContext):
    max_codes = int(cb.data.split("max::", 1)[1])
    data = await state.get_data()
    df = pd.DataFrame(json.loads(data["df_json"]))
    text_col = data["text_col"]
    question = data["question"]

    await cb.message.answer("Генерирую новые категории…")
    await cb.answer()

    texts = [clean_text(x) for x in df[text_col].tolist()]
    await state.update_data(progress_stage="Генерация категорий", progress_done=0, progress_total=len(texts), max_codes=max_codes)

    try:
        codebook = llm_make_codebook(question, texts, max_codes=max_codes)
    except Exception as e:
        await cb.message.answer(f"Ошибка генерации категорий: {e}\nПопробуй ещё раз или пришли категории вручную.")
        await state.set_state(Flow.waiting_codes_choice)
        return

    codebook = ensure_required_codes(codebook)
    await state.update_data(codebook_json=json.dumps([c.model_dump() for c in codebook], ensure_ascii=False))
    await state.set_state(Flow.reviewing_codebook)

    await cb.message.answer(
        "Предлагаемые категории:\n\n"
        + format_codebook(codebook)
        + "\n\nЧто вы хотите сделать?\n"
          "• Нажать ✅ Ок\n"
          "• Или отредактировать командами (help)\n",
        reply_markup=make_review_keyboard(can_regen=True)
    )

# ---------- review/edit ----------

@dp.message(Flow.reviewing_codebook)
async def edit_codebook(msg: Message, state: FSMContext):
    cmd = (msg.text or "").strip()
    if not cmd:
        return

    if cmd.lower() == "help":
        await msg.answer(apply_edit_command([], "help")[1])
        return

    data = await state.get_data()
    if not data.get("codebook_json"):
        await msg.answer("Категории не найдены. Начни заново: /reset")
        return

    codebook = [CodebookItem(**x) for x in json.loads(data["codebook_json"])]
    new_cb, result = apply_edit_command(codebook, cmd)
    await state.update_data(codebook_json=json.dumps([c.model_dump() for c in new_cb], ensure_ascii=False))

    await msg.answer(result + "\n\n" + format_codebook(new_cb),
                     reply_markup=make_review_keyboard(can_regen=bool(data.get("codebook_generated"))))

@dp.callback_query(Flow.reviewing_codebook, F.data == "review::reset")
async def review_reset(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await state.set_state(Flow.waiting_question)
    await cb.message.answer("Процесс перезапущен. Все предыдущие данные были очищены.\n\nВведите вопрос анкеты.")
    await cb.answer()

@dp.callback_query(Flow.reviewing_codebook, F.data == "review::regen")
async def review_regen(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    df = pd.DataFrame(json.loads(data["df_json"]))
    text_col = data["text_col"]
    question = data["question"]
    max_codes = int(data.get("max_codes", 10))

    await cb.message.answer("Перегенерирую категории…")
    await cb.answer()

    texts = [clean_text(x) for x in df[text_col].tolist()]
    try:
        codebook = llm_make_codebook(question, texts, max_codes=max_codes)
    except Exception as e:
        await cb.message.answer(f"Ошибка перегенерации: {e}")
        return

    codebook = ensure_required_codes(codebook)
    await state.update_data(codebook_json=json.dumps([c.model_dump() for c in codebook], ensure_ascii=False))

    await cb.message.answer("Ок, новый список категорий:\n\n" + format_codebook(codebook),
                            reply_markup=make_review_keyboard(can_regen=True))

# ---------- coding ----------

@dp.callback_query(Flow.reviewing_codebook, F.data == "review::ok")
async def review_ok(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    df = pd.DataFrame(json.loads(data["df_json"]))
    text_col = data["text_col"]
    question = data["question"]

    codebook = [CodebookItem(**x) for x in json.loads(data["codebook_json"])]
    codebook = ensure_required_codes(codebook)

    await state.set_state(Flow.coding)
    await state.update_data(cancel_requested=False)

    texts = [clean_text(x) for x in df[text_col].tolist()]
    total = len(texts)

    progress_msg = await cb.message.answer("Начинаю кодировку…")
    await cb.answer()

    await state.update_data(
        progress_stage="Кодировка",
        progress_done=0,
        progress_total=total,
        progress_message_id=progress_msg.message_id,
        progress_chat_id=progress_msg.chat.id,
    )

    out_codes: List[str] = []
    out_comments: List[str] = []

    batches = chunk_list(texts, BATCH_SIZE_FOR_CODING)
    done = 0

    for batch_idx, batch in enumerate(batches, 1):
        st = await state.get_data()
        if st.get("cancel_requested"):
            await bot.edit_message_text(
                chat_id=st["progress_chat_id"],
                message_id=st["progress_message_id"],
                text=f"Отменено ✅ (успели: {done}/{total})\nПришли новый файл или /reset"
            )
            await state.set_state(Flow.waiting_file)
            return

        coded_rows = llm_code_batch(question, batch, codebook)
        for r in coded_rows:
            codes = r.codes[:MAX_CODES_PER_ANSWER] if r.codes else ["Другое/не подходит"]
            out_codes.append("; ".join(codes))
            out_comments.append((r.comment or "").strip())

        done += len(batch)
        await state.update_data(progress_done=done)

        try:
            await bot.edit_message_text(
                chat_id=st["progress_chat_id"],
                message_id=st["progress_message_id"],
                text=f"Кодирую… {done}/{total} (батч {batch_idx}/{len(batches)})\nКоманды: /status, /cancel"
            )
        except Exception:
            pass

        await asyncio.sleep(0.05)

    df_out = df.copy()
    df_out["codes"] = out_codes
    df_out["comment"] = out_comments

    if "row_id" not in df_out.columns and "id" not in df_out.columns:
        df_out.insert(0, "row_id", range(1, len(df_out) + 1))

    meta = pd.DataFrame([{
        "question": question,
        "text_column": text_col,
        "max_codes_per_answer": MAX_CODES_PER_ANSWER,
        "batch_size_for_coding": BATCH_SIZE_FOR_CODING,
        "batch_size_for_codebook": BATCH_SIZE_FOR_CODEBOOK,
        "model": MODEL,
    }])

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="coded")
        cb_df = pd.DataFrame([{"code": c.code, "description": c.description} for c in codebook])
        cb_df.to_excel(writer, index=False, sheet_name="codebook")
        meta.to_excel(writer, index=False, sheet_name="meta")
    bio.seek(0)

    st = await state.get_data()
    try:
        await bot.edit_message_text(
            chat_id=st["progress_chat_id"],
            message_id=st["progress_message_id"],
            text="Готово ✅ Отправляю файл…"
        )
    except Exception:
        pass

    await cb.message.answer_document(
        BufferedInputFile(bio.read(), filename="coded.xlsx"),
        caption="Готово! Лист coded = ответы + коды + комментарии. Лист codebook = категории. Лист meta = контекст."
    )

    await state.clear()
    await state.set_state(Flow.waiting_question)

# ---------- fallback ----------

@dp.message()
async def fallback(msg: Message, state: FSMContext):
    st = await state.get_state()
    if st is None:
        await state.set_state(Flow.waiting_question)
        await msg.answer("Напиши /start чтобы начать.")
        return
    if st == Flow.waiting_file.state:
        await msg.answer("Жду файл .xlsx. Если нужно начать заново — /reset")
    elif st == Flow.waiting_question.state:
        await msg.answer("Пришли текст вопроса анкеты. Если нужно начать заново — /reset")
    else:
        await msg.answer("Сейчас жду следующий шаг. Если зависли — /reset")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
