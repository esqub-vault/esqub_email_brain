import os
from io import BytesIO
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI


# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------

app = FastAPI(
    title="Esqub Email Brain",
    description="LLM microservice to summarise documents and extract key dates.",
    swagger_ui_parameters={"persistAuthorization": True},  # keep auth between calls
)


# -----------------------------------------------------------
# API key security
# -----------------------------------------------------------

API_KEY_HEADER_NAME = "x-api-key"
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def check_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency that validates the x-api-key header against SERVICE_API_KEY.
    - If SERVICE_API_KEY is not set on the server → 500 (misconfiguration).
    - If header missing or mismatched → 401.
    """
    if SERVICE_API_KEY is None:
        # This is a server config issue, not the client's fault.
        raise HTTPException(
            status_code=500,
            detail="Server API key not configured",
        )

    if not api_key or api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

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
# Utility functions
# -----------------------------------------------------------

def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    reader = PdfReader(BytesIO(data))
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            # Ignore pages that fail extraction.
            continue
    return "\n\n".join(texts)


def extract_text_generic(data: bytes, mime_type: Optional[str] = None) -> str:
    """
    Extract text based on mime type.
    You can extend this later for Word, HTML, etc.
    """
    if mime_type == "application/pdf":
        return extract_text_from_pdf_bytes(data)

    if mime_type and mime_type.startswith("text/"):
        # e.g. text/plain, text/markdown
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    # Fallback: try utf-8, then latin-1
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def build_llm():
    """
    Create the ChatOpenAI LLM instance.
    Uses OPENAI_API_KEY from environment (set on Render).
    """
    return ChatOpenAI(
        model="gpt-4.1-mini",   # change later if you want
        temperature=0.2,
    )


def summarise_document(text: str) -> str:
    """Return a markdown summary of the document."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    combined_text = "\n\n".join(chunks)

    prompt = PromptTemplate.from_template(
        "You are helping a very busy business owner.\n"
        "Summarise the following document in clear bullet points.\n"
        "Focus on obligations, risks, money-related terms, dates and key decisions.\n\n"
        "{text}"
    )

    llm = build_llm()
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": combined_text})
    return summary


def extract_key_dates(text: str) -> List[Dict[str, Any]]:
    """Return a list of important dates as structured JSON."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    combined_text = "\n\n".join(chunks)

    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template=(
            "You are an assistant extracting important dates from a document.\n"
            "Return a JSON list where each item has:\n"
            "- raw_text: the exact date text from the document\n"
            "- date_iso: the date in YYYY-MM-DD format if you can infer it, else null\n"
            "- label: a short label like 'Contract start', 'Contract end', 'Notice period', etc.\n"
            "- context: one sentence describing what this date refers to.\n\n"
            "Document:\n{doc}\n\n"
            "{format_instructions}"
        ),
        input_variables=["doc"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = build_llm()
    chain = prompt | llm | parser
    result = chain.invoke({"doc": combined_text})
    return result


# -----------------------------------------------------------
# Simple health endpoint (optional)
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
    api_key: str = Depends(check_api_key),  # enforce x-api-key
):
    """
    Download the file from file_url, extract text, and run LLM processing.
    Returns summary + key dates depending on request_type.
    """
    # 1) Download file from URL
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
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    # 3) Run summarisation / key dates based on request_type
    summary: Optional[str] = None
    key_dates: Optional[List[Dict[str, Any]]] = None

    if req.request_type in ("summary", "summary_and_dates"):
        summary = summarise_document(text)

    if req.request_type in ("key_dates", "summary_and_dates"):
        key_dates = extract_key_dates(text)

    # For now, token usage is not pulled from OpenAI; set zeros.
    usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return {
        "summary_markdown": summary,
        "key_dates": key_dates,
        "llm_model": "gpt-4.1-mini",
        "token_usage": usage.dict(),
    }
