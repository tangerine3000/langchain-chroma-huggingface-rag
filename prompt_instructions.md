# FLAN-T5-Base Prompt Guide

This file defines practical prompts and prompt patterns for `google/flan-t5-base` in this RAG project.

## Model Profile

- Model: `google/flan-t5-base`
- Strengths: concise instruction following, QA, summarization, structured prompts
- Limits: shorter context window than large chat models, weaker long-chain reasoning, can hallucinate if prompt is loose

## Core Prompt Rules

1. Keep instructions explicit and short.
2. Put important rules before context.
3. Use one clear task per prompt.
4. Ask for concise, factual output.
5. In RAG mode, force grounding in retrieved context.
6. In low-confidence cases, require "I do not know based on the documents."

## RAG Prompt Templates

### Prompt: Retrieval QA

Use when relevant chunks are found.

```text
You are a careful retrieval assistant.
Use ONLY the provided context to answer.
If the context is insufficient, say: I do not know based on the documents.
Keep the answer concise and factual.

Context:
{context}

Question: {question}
Answer:
```

### Prompt: Direct QA Fallback

Use when no relevant chunks are found.

```text
You are a helpful assistant.
Answer clearly and concisely.
If uncertain, say you are unsure instead of making up facts.

Question: {question}
Answer:
```

### Prompt: Summary from Context

Use when user asks to summarize retrieved passages.

```text
Summarize the context in 5 to 8 bullet points.
Do not add information not present in the context.

Context:
{context}

Summary:
```

## Prompt Construction Best Practices

- Put instruction block first, then context, then question.
- Use labels exactly: `Context:`, `Question:`, `Answer:`.
- Avoid very long context dumps; keep top relevant chunks only.
- Prefer 3 to 6 chunks instead of many weak chunks.

## Recommended Generation Settings

For factual QA with this model, start with:

- `max_new_tokens=120` to `180`
- `do_sample=False`
- `temperature=0.0` (or omit when sampling disabled)

Example:

```python
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150,
    do_sample=False,
)
```

## Quality Checklist

Before returning an answer:

1. Is the response grounded in retrieved context?
2. Is the answer concise and directly tied to the question?
3. Did the model avoid unsupported claims?
4. If context was weak, did it say it does not know?

## Integration Notes for rag.py

- Keep retrieval-first flow.
- Build prompts through reusable prompt functions.
- Add source/page metadata to chunks for citation-ready answers.
- Consider MMR retrieval and reranking for better relevance.
