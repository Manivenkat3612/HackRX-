import os
import uuid
import tempfile
import time
import fitz  # PyMuPDF
import requests
import json
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
assert GOOGLE_API_KEY and PINECONE_API_KEY and AUTH_TOKEN and PINECONE_INDEX_NAME, "Missing environment variables"

genai.configure(api_key=GOOGLE_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
security = HTTPBearer()
app = FastAPI(title="LLM-Powered Query Retrieval System (HackRx)")

# ---------------------- Auth Security ----------------------
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# ---------------------- Models ----------------------
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ---------------------- Document Parsing ----------------------
def extract_chunks_from_pdf(url: str, chunk_size=800, overlap=200, max_pages=50) -> List[str]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")

    chunks = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text("text", flags=11)
        if not text.strip():
            continue
        for start in range(0, len(text), chunk_size - overlap):
            chunk = text[start:start + chunk_size]
            if len(chunk.strip()) >= 100:
                chunks.append(chunk.strip())
    doc.close()
    if not chunks:
        raise HTTPException(status_code=500, detail="Empty document or failed parsing.")
    return chunks

# ---------------------- Retrieval + Generation ----------------------
def get_contexts(questions: List[str], chunks: List[str], ns: str) -> List[str]:
    contexts = []
    for q in questions:
        q_embedding = embedding_model.encode(q).tolist()
        res = index.query(vector=q_embedding, top_k=10, namespace=ns)
        matches = res.get("matches", [])
        if not matches:
            contexts.append("No relevant content found.")
            continue
        idxs = [int(m["id"].split("_")[1]) for m in matches]
        cands = [chunks[i] for i in idxs if i < len(chunks)]
        scores = reranker_model.predict([[q, c] for c in cands])
        top_chunks = [c for _, c in sorted(zip(scores, cands), reverse=True)[:3]]
        contexts.append("\n---\n".join(top_chunks))
    return contexts

def ask_llm(questions: List[str], contexts: List[str]) -> List[str]:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    answers = []

    for q, ctx in zip(questions, contexts):
        prompt = f"""Answer the following question based strictly on the context:
Context:
{ctx}

Question: {q}
Answer:"""
        try:
            response = model.generate_content(prompt)
            answers.append(response.text.strip())
        except Exception:
            answers.append("Answer not available.")
    return answers

# ---------------------- API Endpoint ----------------------

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(req: QueryRequest, token: str = Depends(verify_token)):
    ns = str(uuid.uuid4())
    chunks = extract_chunks_from_pdf(req.documents)
    embeddings = embedding_model.encode(chunks, batch_size=64).tolist()
    vectors = [(f"vec_{i}", vector) for i, vector in enumerate(embeddings)]
    index.upsert(vectors=vectors, namespace=ns, batch_size=100)

    # Wait briefly for index to reflect upserts
    start = time.time()
    expected = len(vectors)
    while time.time() - start < 5:
        stats = index.describe_index_stats(namespace=ns)
        if stats['namespaces'].get(ns, {}).get('vector_count', 0) >= expected:
            break
        time.sleep(0.5)

    with ThreadPoolExecutor() as pool:
        contexts = pool.submit(get_contexts, req.questions, chunks, ns).result()
        answers = pool.submit(ask_llm, req.questions, contexts).result()

    return QueryResponse(answers=answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
