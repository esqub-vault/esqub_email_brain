import os
import json
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------

app = FastAPI(
    title="Esqub Email Brain",
    description="LLM microservice to summarise documents and extract key dates.",
    swagger_ui_parameters={"persistAuthorization": True},
)

# -----------------------------------------------------------
# API key security (Render service-to-service auth)
# -----------------------------------------------------------

API_KEY_HEADER_NAME = "x-api-key"
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def check_api_key(api_key: str = Security(api_key_header)):
    if SERVICE_API_KEY is None:
        raise HTTPException(status_code=500, detail="Server API key not configured (SERVICE_API_KEY missing).")
    if not api_key or api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key


# -----------------------------------------------------------
# Request / response models
# -----------------------------------------------------------

class ProcessRequest(BaseModel):
    file_url: str
    request_type: str  # "summary", "key_dates", or "summary_and_dates"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# -----------------------------------------------------------
# Settings / guardrails
# -----------------------------------------------------------

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))

# Chunking: keep these conservative to avoid huge prompts
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Hard guardrail on extracted text size to avoid runaway costs/time
MAX_EXTRACTED_CHARS = int(os.environ.get("MAX_EXTRACTED_CHARS", "250000"))  # ~250k chars


def build_llm() -> ChatOpenAI:
    # Uses OPENAI_API_KEY from environment (set in Render)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


# -----------------------------------------------------------
# Utility: PDF/text extraction
# -----------------------------------------------------------

def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf.
    Detect encrypted/password-protected PDFs and return a clean error.
    """
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    # Password-protected detection
    if getattr(reader, "is_encrypted", False):
        # pypdf sometimes allows decrypt attempt; we don't have password, so fail cleanly
        raise HTTPException(
            status_code=422,
            detail="PDF is password-protected/encrypted. Please upload an unlocked PDF."
        )

    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(texts)


def extract_text_generic(data: bytes, mime_type: Optional[str] = None) -> str:
    if mime_type == "application/pdf":
        return extract_text_from_pdf_bytes(data)

    if mime_type and mime_type.startswith("text/"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def _usage_from_langchain_response(resp: Any) -> TokenUsage:
    usage_meta = getattr(resp, "usage_metadata", {}) or {}
    prompt_tokens = usage_meta.get("prompt_tokens") or usage_meta.get("input_tokens") or 0
    completion_tokens = usage_meta.get("completion_tokens") or usage_meta.get("output_tokens") or 0
    total_tokens = usage_meta.get("total_tokens") or (prompt_tokens + completion_tokens)
    return TokenUsage(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(total_tokens),
    )


# -----------------------------------------------------------
# LLM helpers: chunked summary + chunked key-date extraction
# -----------------------------------------------------------

def summarise_document(text: str) -> Tuple[str, TokenUsage]:
    """
    Map-reduce summarisation:
      1) split into chunks
      2) summarise each chunk (MAP)
      3) summarise the summaries (REDUCE)
    This avoids sending the whole document in a single prompt.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)

    llm = build_llm()

    map_prompt = PromptTemplate.from_template(
        "You are helping a very busy business owner.\n"
        "Summarise the following part of a document in clear bullet points.\n"
        "Focus on obligations, risks, money-related terms, dates and key decisions.\n\n"
        "{text}"
    )

    partial_summaries: List[str] = []
    total = TokenUsage()

    for ch in chunks:
        formatted = map_prompt.format(text=ch)
        resp = llm.invoke(formatted)

        part = (getattr(resp, "content", "") or "").strip()
        if part:
            partial_summaries.append(part)

        u = _usage_from_langchain_response(resp)
        total.prompt_tokens += u.prompt_tokens
        total.completion_tokens += u.completion_tokens
        total.total_tokens += u.total_tokens

    if not partial_summaries:
        raise HTTPException(status_code=422, detail="No summary could be produced from extracted text.")

    reduce_prompt = PromptTemplate.from_template(
        "Combine the following partial summaries into one concise markdown summary.\n"
        "Use headings and bullet points.\n"
        "Highlight: obligations, risks, money terms, key dates, and decisions.\n\n"
        "{summaries}"
    )

    reduce_text = "\n\n---\n\n".join(partial_summaries)
    reduce_resp = llm.invoke(reduce_prompt.format(summaries=reduce_text))
    final_summary = (getattr(reduce_resp, "content", "") or "").strip()

    u = _usage_from_langchain_response(reduce_resp)
    total.prompt_tokens += u.prompt_tokens
    total.completion_tokens += u.completion_tokens
    total.total_tokens += u.total_tokens

    if not final_summary:
        raise HTTPException(status_code=422, detail="Final summary was empty.")

    return final_summary, total


def extract_key_dates(text: str) -> Tuple[List[Dict[str, Any]], TokenUsage]:
    """
    Chunked key-date extraction:
      - run extraction per chunk
      - merge arrays
      - de-duplicate
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    llm = build_llm()

    format_instructions = (
        "Return ONLY valid JSON. The top-level value MUST be a JSON array of objects.\n"
        "Each object MUST have keys: raw_text, date_iso, label, context.\n"
        "- raw_text: exact date text from the document\n"
        "- date_iso: YYYY-MM-DD if inferable else null\n"
        "- label: short label e.g., 'Contract start', 'Contract end', 'Payment due', 'Notice period'\n"
        "- context: one sentence about what the date refers to\n"
    )

    all_items: List[Dict[str, Any]] = []
    total = TokenUsage()

    for ch in chunks:
        prompt_text = (
            "You are extracting important dates from a document section.\n"
            "Find contractual dates, renewal dates, notice periods, payment due dates, deadlines.\n\n"
            f"Section:\n{ch}\n\n"
            f"{format_instructions}"
        )

        resp = llm.invoke(prompt_text)
        raw = getattr(resp, "content", "") or ""

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        all_items.append(item)
        except Exception:
            # If the model returns invalid JSON, ignore that chunk instead of failing whole run
            pass

        u = _usage_from_langchain_response(resp)
        total.prompt_tokens += u.prompt_tokens
        total.completion_tokens += u.completion_tokens
        total.total_tokens += u.total_tokens

    # De-duplicate
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in all_items:
        key = (it.get("date_iso"), it.get("label"), it.get("raw_text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    return deduped, total


# -----------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "message": "Esqub Email Brain is running"}


# -----------------------------------------------------------
# Main processing endpoint (protected by API key)
# -----------------------------------------------------------

@app.post("/process-document")
def process_document(
    req: ProcessRequest,
    api_key: str = Depends(check_api_key),
):
    """
    Download file from file_url, extract text, run LLM processing.
    Returns summary + key dates depending on request_type.
    """
    # Validate request_type
    if req.request_type not in ("summary", "key_dates", "summary_and_dates"):
        raise HTTPException(status_code=400, detail="request_type must be: summary, key_dates, summary_and_dates")

    # 1) Download file
    try:
        r = requests.get(req.file_url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {e}")

    content_type = r.headers.get("Content-Type", None)
    data = r.content

    # 2) Extract text
    text = extract_text_generic(data, mime_type=content_type)
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from document.")

    # Guardrail: prevent huge extracted text from creating long processing
    if len(text) > MAX_EXTRACTED_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Document text is too large to process safely right now (chars={len(text)}). "
                   f"Please upload a smaller document or enable stronger chunking/async processing."
        )

    # 3) Run tasks
    summary: Optional[str] = None
    key_dates: Optional[List[Dict[str, Any]]] = None

    total_usage = TokenUsage()

    try:
        if req.request_type in ("summary", "summary_and_dates"):
            summary, u = summarise_document(text)
            total_usage.prompt_tokens += u.prompt_tokens
            total_usage.completion_tokens += u.completion_tokens
            total_usage.total_tokens += u.total_tokens

        if req.request_type in ("key_dates", "summary_and_dates"):
            key_dates, u = extract_key_dates(text)
            total_usage.prompt_tokens += u.prompt_tokens
            total_usage.completion_tokens += u.completion_tokens
            total_usage.total_tokens += u.total_tokens

    except HTTPException:
        # Re-raise our clean errors (422/413/etc.)
        raise
    except Exception as e:
        # If OpenAI returns 429 etc. it may bubble up here depending on libs/versions
        # We convert to a cleaner response
        msg = str(e)
        if "RateLimitError" in msg or "Error code: 429" in msg or "rate_limit" in msg:
            raise HTTPException(
                status_code=429,
                detail=f"LLM provider rate limit / request too large. {msg}"
            )
        raise HTTPException(status_code=500, detail=f"Unexpected processing error: {msg}")

    return {
        "summary_markdown": summary,
        "key_dates": key_dates,
        "llm_model": OPENAI_MODEL,
        "token_usage": total_usage.dict(),
    }
