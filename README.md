
## YTRAG — Yet Another Retrieval-Augmented Generator

This repository contains a small Retrieval-Augmented Generation (RAG) prototype built with LangChain building blocks, FAISS for vector search, and Sentence-Transformers for dense embeddings. The project loads documents (PDF, TXT, CSV, DOCX, JSON, XLSX), splits and embeds them, stores vectors in a FAISS store, and offers a simple search+summarize flow using a Groq LLM wrapper (`langchain_groq`).

This README documents the code structure, how to set up the environment, how to build and query the vectorstore, and notes and suggestions for improvements.

## Table of contents

- Project overview
- Quick start
- Project layout
- Detailed file descriptions
- How the pipeline works (contract + edge cases)
- Troubleshooting & notes
- Next steps and suggestions

## Quick start

Prerequisites

- Python 3.12+ (pyproject specifies >=3.12)
- Recommended: create and activate a virtual environment

On Windows PowerShell (example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: installing `faiss-cpu` on Windows can be tricky. If pip install fails, consult faiss-cpu installation docs or use conda where it is easier to install prebuilt packages.

Set environment variables

- If you will use the Groq LLM integration, set your Groq API key (the code expects a variable but currently initializes with an empty string):

```powershell
setx GROQ_API_KEY "your_api_key_here"
```

### Build the FAISS vector store (one-time)

This project stores a built FAISS index in `faiss_store/faiss.index` along with metadata in `faiss_store/metadata.pkl`.

Run the vector store build pipeline directly:

```powershell
python src\vectorstore.py
```

Or use the example in `src/search.py` which will call `FaissVectorStore.build_from_documents()` when the index is missing.

### Query and summarize (example)

Run the search script interactively or as a script:

```powershell
python src\search.py
```

Or in Python:

```python
from src.search import RAGSearch
rag = RAGSearch()
print(rag.search_and_summarize("What is attention mechanism?", top_k=3))
```

## Project layout

Top-level files

- `main.py` — tiny entrypoint (prints a greeting).
- `pyproject.toml` — project metadata and dependencies.
- `requirements.txt` — pip-compatible dependency list (useful for pip installs).
- `README.md` — this file.

Data & stores

- `data/` — input data. Subfolders: `pdf/`, `text_files/` (contains `machine_learning.txt`, `python_intro.txt`).
- `vector_store/` — contains a Chroma store and sqlite file.
- `faiss_store/` — contains `faiss.index` (the built FAISS index) and should contain `metadata.pkl`.

Source (src/)

- `src/data_loader.py` — loads documents from `data/` using various LangChain community loaders (PDF, TXT, CSV, DOCX, XLSX, JSON). Returns LangChain Document objects.
- `src/embedding.py` — defines an `EmbeddingPipeline` that splits documents using `RecursiveCharacterTextSplitter` and generates sentence-transformer embeddings.
- `src/vectorstore.py` — defines `FaissVectorStore` which builds a FAISS index from embeddings, saves/loads index and metadata, and exposes query/search methods.
- `src/search.py` — defines `RAGSearch` which wires the vectorstore and a Groq-backed LLM (`ChatGroq`) and provides `search_and_summarize()`.
- `src/_init_.py` — package init placeholder (empty).

## Detailed file descriptions

- `src/data_loader.py`
	- Purpose: discover and load documents inside the `data/` folder into LangChain Document objects.
	- Supported formats: PDF, TXT, CSV, XLSX, DOCX, JSON.
	- Behavior: prints debug information and returns a flat list of loaded documents.
	- Notes: error handling prints the exception and continues. Large directories or malformed files may slow loading.

- `src/embedding.py`
	- Purpose: split documents into chunks (default size 1000 chars with 200 overlap) and generate embeddings with a Sentence-Transformers model (default `all-MiniLM-L6-v2`).
	- Exposes: `EmbeddingPipeline.chunk_documents(documents)` and `embed_chunks(chunks)`.
	- Notes: uses `SentenceTransformer.encode(...)` which returns numpy arrays; ensure the model is installed and local resources are sufficient.

- `src/vectorstore.py`
	- Purpose: manage FAISS index lifecycle: build from documents, add embeddings, save to disk, load from disk, and query.
	- Important methods: `build_from_documents`, `add_embeddings`, `save`, `load`, `search`, `query`.
	- Storage: writes `faiss.index` and `metadata.pkl` into `faiss_store/` by default.
	- Notes: the store assumes all embeddings share the same dimension. Metadata is stored as a list aligned with index ids.

- `src/search.py`
	- Purpose: high-level RAG helper that ensures the vectorstore is built/loaded and provides `search_and_summarize(query, top_k)`.
	- LLM integration: uses `langchain_groq.ChatGroq`. The current code passes an empty string as `groq_api_key` — you should populate this from an environment variable for real use.

## Contract (tiny)

- Inputs: local `data/` folder populated with supported documents; optional environment variables (GROQ_API_KEY).
- Outputs: FAISS index files in `faiss_store/`, and textual summaries returned by `RAGSearch.search_and_summarize()`.
- Error modes: loader catches and logs individual file load errors; building index raises if embedding model or faiss are unavailable.

## Edge cases & recommendations

- Empty data folder: the vectorstore builder will create an index with zero vectors; querying will fail or return empty results. Add a guard to raise a readable error when no documents are found.
- Missing dependencies: `faiss-cpu`, `sentence-transformers`, `pymupdf`, and other native dependencies can be problematic on some platforms. Recommend using a conda environment or Dockerfile for reproducible setup.
- Model size and memory: Some sentence-transformers models can be large. The default small model is reasonable for local dev. For production, consider vector quantization or an external embedding service.
- Metadata alignment: metadata list is appended in the same order as vectors are added. If vectors are removed or reindexed, metadata must be handled carefully.
- Concurrency: FAISS indexes are not thread-safe for write operations. If this project will support concurrent writes, add locking or use an external vector DB.

## Troubleshooting

- FAISS import errors on Windows:
	- Option 1: use conda and install `faiss-cpu` from conda-forge.
	- Option 2: use an alternative vector store like Chroma (already present in `vector_store/` folder) or use a cloud vector DB.
- Embedding model download stalls or OOMs:
	- Use a lighter sentence-transformers model, increase swap, or run on a machine with more RAM/GPU.
- Groq LLM not responding or failing: supply a valid `GROQ_API_KEY` and check `langchain_groq` configuration.

## Development notes & suggested improvements

- Move configuration (persist directories, model names, API keys) into a single config module or `.env` and read with `python-dotenv` (the repo already imports `load_dotenv()` in `src/search.py`).
- Add unit tests: small tests for `data_loader.load_all_documents()` (mock file system), `EmbeddingPipeline.chunk_documents()` and `FaissVectorStore.add_embeddings()`.
- Add CLI: create a tiny CLI (click/typer) for building the index, listing documents, and running queries.
- Add typed interfaces and docstrings: expand docstrings and add type hints where missing to improve readability and maintainability.
- Add graceful handling for missing index when querying and better logging (use `logging` instead of prints).

## How to contribute

- Fork the repository, create a topic branch, add features or bug fixes, add tests, and open a PR. Focus areas: reproducible environment (Docker/conda), tests, and configuration consolidation.

## Summary

This repository contains a small RAG prototype that demonstrates loading documents, producing embeddings with Sentence-Transformers, storing vectors in FAISS, and using a Groq-backed LLM to summarize search results. The code is a solid starting point — follow the suggestions above to make it more robust, testable, and production-ready.

#   p r o j e c t - R A G  
 