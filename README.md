# MP2 · Mini-RAG — Ask the Sherlock Holmes Stories

A small end-to-end RAG pipeline: load five Sherlock Holmes story summaries,
chunk them, embed the chunks, store them in Qdrant, then answer questions
with citations using only the retrieved text.

```
corpus/*.txt → chunks → embeddings → Qdrant
                                        ↓
                    question → retrieve (top-k) → answer + citations
```

## Project layout

```
learner/
├── mp2_rag.py              # the pipeline (ingest / ask / validate)
├── mp2_reflection.md       # design decisions, results, experiments
├── mp2_validation.txt      # saved output of `validate`
├── MP2_FAQ.md              # background Q&A
├── requirements.txt
├── .env.example            # copy to .env and fill in
├── corpus/                 # 5 story summaries
└── data/
    ├── predefined_questions.jsonl   # 2 provided golden questions
    └── learner_questions.jsonl      # 3 questions of my own
```

## Stack and settings

| Part | Choice |
|---|---|
| Embeddings | `text-embedding-3-small` (1536-dim, cosine) |
| Chat model | `gpt-4o-mini`, temperature 0 |
| Vector store | Qdrant, collection `mp2_sherlock` |
| Chunking | Paragraphs packed up to 500 characters; oversized paragraphs split with 80-character overlap; short unpunctuated lines start a new section |
| Retrieval | top-k = 3 |
| Prompt | Answer only from the excerpts, refuse if the answer isn't there, cite story and section |

## Running it

Run everything from `learner/`.

**1. Configure.** Copy `.env.example` to `.env` and set plain `KEY=value`
lines (no `export`). The script loads `.env` automatically.

| Variable | Notes |
|---|---|
| `OPENAI_API_KEY` | Your API key |
| `OPENAI_BASE_URL` | Only for proxy keys (for example `https://openai.vocareum.com/v1`). Without it a proxy key gives a 401 |
| `QDRANT_URL` | Qdrant Cloud URL, or `http://localhost:6333` for local Docker |
| `QDRANT_API_KEY` | Leave empty for local Qdrant |

Never submit or commit `.env`.

**2. Install and run.**

```bash
pip install -r requirements.txt
python mp2_rag.py ingest      # once: chunk, embed, store (wipes and rebuilds the collection)
python mp2_rag.py validate    # run the golden and own questions
python mp2_rag.py ask         # interactive Q&A; an empty line exits
```

Save evidence with `python mp2_rag.py validate > mp2_validation.txt`.

## Interpreting results

**`ingest`:** expect about 70–90 chunks (this build: 89 = 14 / 18 / 15 / 20 / 22).
Fewer than 30 or more than 200 suggests a chunking bug. `✓ Done` means all
chunks were embedded and stored.

**`validate`:**

| Field | Meaning |
|---|---|
| `✓` / `✗` | The expected story file is, or is not, among the cited sources. This is the pass/fail check |
| `Facts matched n/m` | How many `expected_facts` strings appear in the answer (case-insensitive substring). A diagnostic only |
| `Latency` | Retrieve plus generate time; about 3–4 s is normal |
| `Source-match x/y` | Number of ✓ per question file |

How to read it:

- **✓ with low facts matched** is common. Either the answer paraphrases
  ("snake" instead of "swamp adder"), or retrieval found the right story but
  not the right section. Compare facts in the retrieved chunks against facts in
  the answer to tell which.
- **✗** means the right story wasn't retrieved. Inspect the chunks, reword the
  question, or raise `k`.
- **"The excerpts do not contain the answer."** is the intended refusal. Treat
  it as a retrieval miss, not a model error.
- Citations list only chunks the answer names, falling back to all retrieved
  chunks if it names none.

## Results

Source-match is 5/5 (2 provided questions + 3 of my own, one per remaining story).
Facts matched is 1–3 out of 5–6 per question. Retrieval finds the right story
every time but often misses the right section; k=5 helps some questions (q2,
q4) but not others. Details, per-question numbers and the k=3 vs k=5
comparison are in `learner/mp2_reflection.md`; the raw run is in
`learner/mp2_validation.txt`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RAG configuration error` on start | A required `.env` value is missing or still a placeholder |
| `401 Incorrect API key` | Missing `OPENAI_BASE_URL` for a proxy key, or a wrong key |
| `Service unavailable` from Qdrant | Free cluster was asleep; wait about 60 s and retry |
| Empty or unrelated answers | Run `ingest` first; the collection may be empty |
| Garbled ✓/✗ on Windows | The script sets UTF-8; if it persists, set `PYTHONIOENCODING=utf-8` |

## Corpus

Condensed summaries, in original wording, of five stories from Arthur Conan
Doyle's *The Adventures of Sherlock Holmes* (1892, public domain):
`01_red_headed_league`, `02_speckled_band`, `03_blue_carbuncle`,
`04_engineers_thumb`, `05_scandal_in_bohemia`.
