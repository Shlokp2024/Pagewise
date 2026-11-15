import os
import tempfile
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_database(pdf_bytes, embeddings):
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with open(path, "wb") as handle:
            handle.write(pdf_bytes)
        try:
            pages = PyPDFLoader(path).load()
        except Exception as error:
            raise ValueError("This PDF could not be read. It may be damaged or password protected.") from error
    finally:
        if os.path.exists(path):
            os.remove(path)

    readable = [page for page in pages if page.page_content.strip()]
    if not readable:
        raise ValueError("No text found in this PDF. Scanned documents are not supported yet.")

    chunks = splitter.split_documents(readable)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"doc_{uuid.uuid4().hex}",
    )
    return store, len(pages), len(chunks)