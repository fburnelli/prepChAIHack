
import time
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def generate_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Free and fast
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def generate_vector_store():
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                      persist_directory="./chroma_store"))
    collection = client.create_collection(name="legal_vector")
    return
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"doc_{i}"]
       )


if __name__ == "__main__":
    print("Generating the vector store...")
    start_time = time.time()  # salva il tempo iniziale
    # generate_vector_store()  # qui chiami la funzione reale
    end_time = time.time()    # salva il tempo finale
    elapsed = end_time - start_time

    print(f"Vector store Done in {elapsed:.2f} seconds")