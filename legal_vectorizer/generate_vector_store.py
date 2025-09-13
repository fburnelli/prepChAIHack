import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

model = SentenceTransformer("all-MiniLM-L6-v2")
import chromadb



def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_vector_store(data_folder="data"):

    client = chromadb.PersistentClient(path="./chroma_db", settings=Settings())
    collection = client.get_or_create_collection(name="pdf_vectors")
    
    
    
    # Loop through all PDFs in the folder
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            print(f"📄 Processing: {filename}")

            # Extract and chunk text
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)

            # Generate embeddings
            embeddings = model.encode(chunks, show_progress_bar=True)

            # Add to ChromaDB
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                collection.add(documents=[chunk], embeddings=[embeddings[i].tolist()], ids=[doc_id])

    return collection
        


def query(collection):
    query_text = "Ware die Daten anonimisiert?"
    query_embedding = model.encode([query_text])

    results = collection.query(
    query_embeddings=query_embedding,
    n_results=1,
    include=["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Risultati per la query: \"{query_text}\"\n")

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        doc_id = results["metadatas"][0][i]
        distance = results["distances"][0][i]
    
        print(f"📄 Documento {i+1} (ID: {doc_id})")
        print(f"🔢 Similarità: {1 - distance:.2f}")
        print(f"📝 Testo:\n{doc[:200]}...\n")  

if __name__ == "__main__":
    collection=generate_vector_store()
    query(collection)
