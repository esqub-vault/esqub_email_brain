from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
from typing import Optional, List, Dict, Any

from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI


app = FastAPI(title="Esqub Email Brain")


class ProcessRequest(BaseModel):
    file_url: str
    request_type: str  # "summary", "key_dates", "summary_and_dates"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Simple PDF text extractor using pypdf.
    """
    reader = PdfReader(BytesIO(data))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(texts)


def extract_text_generic(data: bytes, mime_type: Optional[str] = None) -> str:
    """
    Decide how to extract text based on mime type.
    """
    if mime_type == "application/pdf":
        return extract_text_from_pdf_bytes(data)

    if mime_type == "text/plain":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    # fallback
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def build_llm():
    """
    Configure the LLM here.
    """
    return ChatOpenAI(
        model="gpt-4.1-mini",  # you can change model later
        temperature=0.2,
    )


def summarise_document(text: str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    combined_text = "\n\n".join(chunks)

    prompt = PromptTemplate.from_template(
        "You are helping a busy business owner.\n"
        "Summarise the following document in clear bullet points.\n"
        "Focus on obligations, risks, money-related terms, dates and key decisions.\n\n"
        "{text}"
    )

    llm = build_llm()
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": combined_text})
    return summary


def extract_key_dates(text: str) -> List[Dict[str, Any]]:
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
            "- date_iso: the date in YYYY-MM-DD format if you can guess it, else null\n"
            "- label: short label like 'Contract start', 'Contract end', 'Notice period', etc.\n"
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


@app.post("/process-document")
def process_document(req: ProcessRequest):
    # 1) Download file from Supabase signed URL
    try:
        r = requests.get(req.file_url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {e}")

    content_type = r.headers.get("Content-Type", None)
    data = r.content

    # 2) Extract text
    text = extract_text_generic(data, mime_type=content_type)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    # 3) Run summarisation / key dates based on request_type
    summary: Optional[str] = None
    key_dates: Optional[List[Dict[str, Any]]] = None

    if req.request_type in ("summary", "summary_and_dates"):
        summary = summarise_document(text)

    if req.request_type in ("key_dates", "summary_and_dates"):
        key_dates = extract_key_dates(text)

    usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return {
        "summary_markdown": summary,
        "key_dates": key_dates,
        "llm_model": "gpt-4.1-mini",
        "token_usage": usage.dict(),
    }
