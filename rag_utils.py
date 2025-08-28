import fitz  # PyMuPDF for PDFs
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import textwrap

# Initialize models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# 1. Extract text from uploaded PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 2. Chunk the extracted text
def chunk_text(text, chunk_size=200):
    return textwrap.wrap(text, width=chunk_size)

# 3. Build FAISS index dynamically
def build_faiss(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

# 4. Retrieve & generate answer dynamically
def retrieve_and_answer(query, index, chunks, top_k=4):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    retrieved_texts = [chunks[i] for i in indices[0]]
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = qa_pipeline(prompt, max_length=150, do_sample=False)
    return result[0]['generated_text']
