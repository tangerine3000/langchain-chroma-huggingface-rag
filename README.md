
# LangChain + Chroma Retrieval-First RAG

This project is a local Retrieval-Augmented Generation pipeline with:

- PDF ingestion via PyPDF
- Embeddings via Hugging Face (`all-MiniLM-L6-v2`)
- Persistent Chroma vector store
- Retrieval-first answer flow with direct-answer fallback
- Prompt templates loaded from `prompt_instructions.md`

## Project Structure

- [rag.py](rag.py): Main script for indexing, retrieval, and answering.
- [prompt_instructions.md](prompt_instructions.md): Prompt templates used at runtime.
- [requirements](requirements): Python dependencies list.
- [knowledge_base_pdfs](knowledge_base_pdfs): Local PDF folder used for indexing.
- [chroma_db](chroma_db): Persisted Chroma database.

## Setup

Install dependencies:

```bash
py -m pip install -r requirements
```

## Configuration

Edit constants in [rag.py](rag.py):

- `DOCS_PATH`: folder containing source PDFs.
- `REBUILD_INDEX`: controls re-indexing behavior.
- `RETRIEVE_K`: number of candidate chunks to retrieve.
- `MAX_DISTANCE`: retrieval distance threshold (smaller = stricter relevance).

## Rebuild Index

Set `REBUILD_INDEX = True` in [rag.py](rag.py) when you add or change PDFs.

Then run:

```bash
py rag.py
```

After rebuild completes, set `REBUILD_INDEX = False` for normal query runs.

## Run

For normal usage (reuse existing Chroma index):

1. Ensure `REBUILD_INDEX = False` in [rag.py](rag.py).
2. Run:

```bash
py rag.py
```

3. Enter a question when prompted.

## Prompt Templates

Runtime prompts are loaded from [prompt_instructions.md](prompt_instructions.md):

- `Prompt: Retrieval QA`
- `Prompt: Direct QA Fallback`

If these sections are missing, [rag.py](rag.py) falls back to built-in default prompts.

## Troubleshooting

- No PDFs found or file/path errors:
	- Confirm [knowledge_base_pdfs](knowledge_base_pdfs) exists.
	- Confirm `DOCS_PATH` in [rag.py](rag.py) points to a valid folder.
	- Re-run with `REBUILD_INDEX = True` after fixing the path.

- Answers are generic or not grounded:
	- Increase retrieval strictness by lowering `MAX_DISTANCE` in [rag.py](rag.py) (for example, `1.2` to `0.9`).
	- Increase recall by raising `RETRIEVE_K` (for example, `5` to `8`).
	- Verify your `Prompt: Retrieval QA` block in [prompt_instructions.md](prompt_instructions.md) still includes `{context}` and `{question}`.

- First run is very slow:
	- This is expected during first embedding/index build.
	- Keep `REBUILD_INDEX = False` for normal runs to reuse [chroma_db](chroma_db).

- Index seems stale after adding new PDFs:
	- Set `REBUILD_INDEX = True` in [rag.py](rag.py), run once, then set it back to `False`.

- Template changes do not affect output:
	- Ensure headings in [prompt_instructions.md](prompt_instructions.md) match exactly:
		- `### Prompt: Retrieval QA`
		- `### Prompt: Direct QA Fallback`
	- Ensure placeholders remain valid:
		- Retrieval template needs `{context}` and `{question}`.
		- Direct template needs `{question}`.

## Notes

- This is retrieval-first RAG, not a full autonomous agent loop.
- Chroma is persistent (`persist_directory="chroma_db"`), so embeddings are reused across runs.
- If no retrieved chunks pass `MAX_DISTANCE`, the script uses direct-answer fallback.
