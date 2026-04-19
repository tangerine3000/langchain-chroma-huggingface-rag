# Agentic RAG (Retrieval-Augmented Generation) 
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Path to the folder that contains your PDF files.
DOCS_PATH = "knowledge_base_pdfs/"
BATCH_SIZE = 5000
RETRIEVE_K = 5
MAX_DISTANCE = 1.2
REBUILD_INDEX = False
PROMPT_INSTRUCTIONS_FILE = "prompt_instructions.md"

# Default prompt templates
DEFAULT_RETRIEVAL_PROMPT = (
    "You are a careful retrieval assistant.\n"
    "Use ONLY the provided context to answer.\n"
    "If the context is insufficient, say: I do not know based on the documents.\n"
    "Keep the answer concise and factual.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# This prompt is used when no relevant documents are found, so the model must answer based on its own knowledge.
DEFAULT_DIRECT_PROMPT = (
    "You are a helpful assistant.\n"
    "Answer clearly and concisely.\n"
    "If uncertain, say you are unsure instead of making up facts.\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Load PDFs from a folder
def load_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

# Load prompt templates from a markdown file with a specific format. If the file or the prompt is not found, return the default template.
def load_prompt_instructions(prompt_name, default_template):
    if not os.path.exists(PROMPT_INSTRUCTIONS_FILE):
        return default_template

    with open(PROMPT_INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        rf"###\s+Prompt:\s+{re.escape(prompt_name)}\s+.*?```text\n(.*?)\n```",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return default_template

    return match.group(1).strip()

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    collection_name="rag_store",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

# Rebuild the index if needed
if REBUILD_INDEX:
    docs = load_docs(DOCS_PATH)
    print("PDF Pages Loaded:", len(docs))

    # Split PDFs into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = text_splitter.split_documents(docs)
    print("Chunks Created:", len(chunks))

    # Chroma enforces a max upsert batch size; add texts in smaller chunks.
    texts = [c.page_content for c in chunks]
    for i in range(0, len(texts), BATCH_SIZE):
        db.add_texts(texts[i:i + BATCH_SIZE])
    print("Index rebuild complete.")
else:
    print("REBUILD_INDEX is False. Reusing existing Chroma data from 'chroma_db'.")

# Local LLM
llm = pipeline(
    "text2text-generation",              
    model="google/flan-t5-base",
    max_new_tokens=150
)
MODEL_MAX_INPUT_TOKENS = llm.tokenizer.model_max_length
INPUT_TOKEN_RESERVE = 64

# Load prompt templates (with fallback to defaults)
RETRIEVAL_QA_PROMPT = load_prompt_instructions("Retrieval QA", DEFAULT_RETRIEVAL_PROMPT)
DIRECT_QA_PROMPT = load_prompt_instructions("Direct QA Fallback", DEFAULT_DIRECT_PROMPT)

# Retrieve relevant document chunks based on the query. Only return chunks that are within a certain distance threshold to ensure relevance.
def retrieve_context(query):
    results_with_scores = db.similarity_search_with_score(query, k=RETRIEVE_K)
    strong_results = [doc for doc, score in results_with_scores if score <= MAX_DISTANCE]
    return strong_results


# Build a prompt that includes as much relevant context as possible without exceeding the model's input token limit. 
# If no chunks fit, include a truncated version of the most relevant chunk.
def build_retrieval_prompt(query, results):
    max_input_tokens = max(1, MODEL_MAX_INPUT_TOKENS - INPUT_TOKEN_RESERVE)
    selected_chunks = []

    for result in results:
        candidate_context = "\n\n".join(selected_chunks + [result.page_content])
        candidate_prompt = RETRIEVAL_QA_PROMPT.format(context=candidate_context, question=query)
        token_count = len(llm.tokenizer(candidate_prompt, add_special_tokens=True)["input_ids"])
        if token_count > max_input_tokens:
            break
        selected_chunks.append(result.page_content)

    if not selected_chunks and results:
        prompt_without_context = RETRIEVAL_QA_PROMPT.format(context="", question=query)
        prompt_tokens = len(llm.tokenizer(prompt_without_context, add_special_tokens=True)["input_ids"])
        available_context_tokens = max(1, max_input_tokens - prompt_tokens)
        truncated_context_tokens = llm.tokenizer(
            results[0].page_content,
            add_special_tokens=False,
            truncation=True,
            max_length=available_context_tokens,
        )["input_ids"]
        truncated_context = llm.tokenizer.decode(truncated_context_tokens, skip_special_tokens=True).strip()
        if truncated_context:
            selected_chunks.append(truncated_context)

    context = "\n\n".join(selected_chunks)
    return RETRIEVAL_QA_PROMPT.format(context=context, question=query), len(selected_chunks)


# RAG
def rag_answer(query):
    results = retrieve_context(query)

    if results:
        final_prompt, used_chunks = build_retrieval_prompt(query, results)
        print(f"🕵️ Retrieval-first: using {used_chunks} relevant document chunks.")
    else:
        print("🤖 Retrieval-first: no relevant chunks found, falling back to direct answer.")
        final_prompt = DIRECT_QA_PROMPT.format(question=query)

    response = llm(final_prompt, truncation=True)[0]["generated_text"]
    return response


if __name__ == "__main__":
    user_query = input("Ask a question: ").strip()
    if not user_query:
        print("Please provide a non-empty question.")
    else:
        answer = rag_answer(user_query)
        print("\nAnswer:\n", answer)