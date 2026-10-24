# MP2 · Reflection

## 1. What I built

A mini-RAG pipeline over 5 Sherlock Holmes story summaries:
corpus → paragraph/section-aware chunks → OpenAI embeddings → Qdrant → top-k retrieval → grounded answer with citations.
Run with `python mp2_rag_starter.py ingest | ask | validate`.

| Setting | Value |
|---|---|
| Embeddings | `text-embedding-3-small` (1536-dim, cosine) |
| Chat model | `gpt-4o-mini`, temperature 0 |
| Vector store | Qdrant Cloud, collection `mp2_sherlock` |
| Chunking | paragraphs packed up to 500 chars; oversized paragraphs split with 80-char overlap; short unpunctuated lines start a new section |
| Retrieval | top-k = 3 |
| Corpus | provided Sherlock summaries (no substitute corpus) |

## 2. Results

Source-match was 5/5 (2 predefined + 3 of my own). Facts matched is only a diagnostic.

| Question | Source ✓ | Facts matched |
|---|---|---|
| q1 Red-Headed League assistant | ✓ | 2/5 |
| q2 Speckled Band murder weapon | ✓ | 3/6 |
| q3 Blue Carbuncle goose trail | ✓ | 2/5 |
| q4 Engineer's Thumb press secret | ✓ | 1/5 |
| q5 Scandal in Bohemia (multi-hop) | ✓ | 2/5 |

Chunks: 89 total (14 / 18 / 15 / 20 / 22 per story). Latency about 3–4 s per question.

## 3. Design decisions and why

- **Chunking:** paragraph packing keeps each story section together, so a chunk carries its section title as metadata for citations. Fixed windows would have cut mid-sentence.
- **k = 3:** the starter default. Kept for the baseline; see Section 5 for the k = 5 experiment.
- **Prompt:** "use ONLY the excerpts", an exact refusal line, and a `Sources:` line. This is to stop the model falling back on its own memory of the original stories.
- **Citations:** limited to chunks the answer names, with a fallback to all retrieved chunks so citations never vanish.

## 4. What went wrong and what I fixed

- `qdrant.recreate_collection` and `qdrant.search` are deprecated, so I switched to `create_collection` and `query_points`.
- Windows console and file encoding broke the ✓/✗ output, so I set UTF-8.
- First `ingest` failed with a 401: the `.env` key is a Vocareum proxy key and needs `OPENAI_BASE_URL`. It worked once the base URL was present.

## 5. Experiments: is the weakness retrieval or generation?

I printed the retrieved chunks for every question and counted how many expected facts appear in the retrieved text versus in the answer, at k = 3 and k = 5.

| Question | Facts | k=3 in chunks | k=3 in answer | k=5 in chunks | k=5 in answer |
|---|---|---|---|---|---|
| q1 Red-Headed League | 5 | 1 | 2 | 1 | 2 |
| q2 Speckled Band | 6 | 5 | 3 | 5 | 5 |
| q3 Blue Carbuncle | 5 | 2 | 2 | 2 | 2 |
| q4 Engineer's Thumb | 5 | 1 | 1 | 2 | 2 |
| q5 Scandal in Bohemia | 5 | 4 | 2 | 5 | 2 |

Findings:

- **Retrieval is the main bottleneck.** The right story is always found, but the right section often isn't. For q1, "The Identity of the Assistant" (with John Clay) is missing at k=3. For q4, "The Coining Operation" (counterfeiting, Doctor Becher) is missing at k=3 and the model honestly answered that the real name wasn't in the excerpts, which shows the "use only the excerpts" prompt works. For q3, the "James Ryder Confesses" section is never retrieved even at k=5, so the answer misses who hid the stone.
- **k = 5 helps in some cases.** q2 went from 3 to 5 facts and q4 from 1 to 2, and it brought in the Coining Operation chunk and the real name. q1, q3 and q5 did not improve in facts matched.
- **Generation also loses facts when the context is good.** q5 had 4–5 of 5 facts in the chunks but only 2 in the answer. The answer is correct but uses different words (for example no "smoke rocket" or "St Monica"), which the substring check doesn't count.
- **Why retrieval misses:** the question wording matches a different section. "How did the carbuncle end up in a goose" matches the section where it is found, not the confession that explains it. Plain embedding similarity on short chunks is weak for this kind of "explain who did it" question.
- **Facts matched is a harsh metric.** It penalises correct paraphrases, so I treat source-match as the grade-relevant check and facts matched as a diagnostic.

Possible next steps (not done):

- Raise k to 5 as the default, since it costs little on this corpus.
- Hybrid BM25 + RRF and a cross-encoder rerank (the optional stretch). Names like "Ryder" and "Becher" are rare exact terms, so keyword matching could help more than I first expected.
- `gpt-4o` for generation, only if answers stay thin when the right chunks are present.

## 6. What I'd do differently

- Print retrieved chunks from the start instead of trusting the final answer.
- Measure facts per question before and after each change, not just source-match.
- Keep `.env` out of the submission.
