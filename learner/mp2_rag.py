"""MP2 · Mini-RAG
===============

A complete RAG pipeline over the Sherlock Holmes corpus.

Pipeline:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence:
    pip install -r requirements.txt
    (credentials are loaded from .env automatically via python-dotenv)
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # ✓/✗ and em dashes on Windows consoles

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

required_settings = ("OPENAI_API_KEY", "QDRANT_URL")
missing_settings = [name for name in required_settings if not os.environ.get(name)]
placeholder_settings = [
    name for name in (*required_settings, "QDRANT_API_KEY")
    if os.environ.get(name, "").lower().startswith(("your-", "https://your-"))
]
if missing_settings or placeholder_settings:
    problems = []
    if missing_settings:
        problems.append(f"missing: {', '.join(missing_settings)}")
    if placeholder_settings:
        problems.append(f"still using placeholders: {', '.join(placeholder_settings)}")
    raise SystemExit(
        "RAG configuration error (" + "; ".join(problems) + "). "
        "Copy learner/.env.example to learner/.env and add your actual credentials."
    )

openai = OpenAI()
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).
    """
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        documents.append({"source": path.name, "title": title, "text": text})
    return documents


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.

    Paragraphs are packed up to TARGET_CHUNK_SIZE. Short lines without
    terminal punctuation start a new section, and oversized paragraphs are
    split into overlapping windows.
    """
    chunks: list[dict[str, Any]] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", doc["text"]) if part.strip()]
    section = doc["title"]
    pending: list[str] = []

    def add_chunk(text: str, chunk_section: str) -> None:
        if text.strip():
            chunks.append({
                "source": doc["source"],
                "title": doc["title"],
                "section": chunk_section,
                "text": text.strip(),
            })

    def flush_pending() -> None:
        if pending:
            add_chunk("\n\n".join(pending), section)
            pending.clear()

    for paragraph in paragraphs:
        if paragraph == doc["title"]:
            continue

        if len(paragraph) <= 100 and not re.search(r"[.!?][\"')\]]*$", paragraph):
            flush_pending()
            section = paragraph
            continue

        if len(paragraph) > TARGET_CHUNK_SIZE:
            flush_pending()
            step = max(1, TARGET_CHUNK_SIZE - CHUNK_OVERLAP)
            for start in range(0, len(paragraph), step):
                add_chunk(paragraph[start:start + TARGET_CHUNK_SIZE], section)
            continue

        candidate = "\n\n".join([*pending, paragraph])
        if pending and len(candidate) > TARGET_CHUNK_SIZE:
            flush_pending()
        pending.append(paragraph)

    flush_pending()
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).
    """
    if not texts:
        return []
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection with cosine distance."""
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant."""
    if not chunks:
        return
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=chunk)
        for chunk, embedding in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query, each with its similarity score."""
    query_vector = embed_texts([query])[0]
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    ).points
    results = []
    for hit in hits:
        chunk = dict(hit.payload or {})
        chunk["score"] = hit.score
        results.append(chunk)
    return results


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, reply exactly "The excerpts do not contain
the answer." and do not use outside knowledge of the stories. Cite the source
(story title + section) inline, and end with a line "Sources: <title> — <section>"
listing only the excerpts you actually used."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    Returns a dict with: question, answer, citations, latency_ms.
    """
    started = time.perf_counter()
    retrieved = retrieve(question, k=k)
    context = "\n\n".join(
        f"[Source: {chunk.get('title', 'Unknown')} — {chunk.get('section', 'Unknown')}; "
        f"file: {chunk.get('source', 'unknown')}]\n{chunk.get('text', '')}"
        for chunk in retrieved
    )
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ],
    )
    answer_text = response.choices[0].message.content or ""
    # Keep only chunks the answer actually names (by section or title); fall back
    # to every retrieved chunk if the model named none, so citations never vanish.
    answer_lower = answer_text.lower()
    used = [
        c for c in retrieved
        if str(c.get("section", "")).lower() in answer_lower
        or str(c.get("title", "")).lower() in answer_lower
    ]
    citations = []
    seen_citations = set()
    for chunk in used or retrieved:
        citation_key = (chunk.get("source"), chunk.get("section"))
        if citation_key in seen_citations:
            continue
        seen_citations.add(citation_key)
        citations.append({
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "section": chunk.get("section", ""),
            "score": chunk.get("score"),
        })
    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text(encoding="utf-8").splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
