import os
import uuid
import requests
import tempfile
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone
import google.generativeai as genai
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Load environment variables from a .env file for local development
load_dotenv()

# --- Configuration & Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "650a0269eda6ed91714c00834a396c2727072511734daa3d3c7e89e60d17da41")

# Validate that all necessary environment variables are set
if not all([GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise ValueError("One or more required environment variables are not set (GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME).")

# Initialize services using the latest library standards
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize the embedding model for retrieval (fast)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize the Cross-Encoder model for the more accurate re-ranking step
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

app = FastAPI(title="High-Accuracy Intelligent Query-Retrieval System with Re-ranking")
security = HTTPBearer()

# --- Pydantic Models (matching submission spec) ---
class QueryRequest(BaseModel):
    documents: str = Field(..., example="URL to the policy PDF, DOCX, or EML")
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates the bearer token provided in the request header."""
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Core Logic Functions ---
def get_text_chunks_unstructured(doc_url: str) -> List[str]:
    """
    Downloads a document and uses unstructured.io for high-accuracy parsing.
    Uses tempfile for cross-platform compatibility.
    """
    temp_file_path = None
    try:
        response = requests.get(doc_url, timeout=20)
        response.raise_for_status()

        # Use tempfile to create a temporary file that works on all OS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            temp_file_path = tmp.name
        
        # Use unstructured.partition.auto to automatically detect and parse the file type
        elements = partition(filename=temp_file_path, strategy="fast")
        
        # Use unstructured's intelligent chunking to group elements by title and context
        chunks = chunk_by_title(elements, max_characters=1000)
        chunk_texts = [chunk.text for chunk in chunks]

        if not chunk_texts:
            raise ValueError("Document could not be parsed into meaningful chunks.")
        return chunk_texts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document with unstructured.io: {e}")
    finally:
        # Ensure the temporary file is cleaned up
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_answer_from_llm(question: str, context: str) -> str:
    """Generates a concise answer from Google's Gemini model based on the high-quality context."""
    system_prompt = """
    You are an expert AI assistant specializing in analyzing policy documents.
    Your task is to answer the user's question accurately and concisely based ONLY on the provided context.
    Do not use any external knowledge. If the context does not contain the answer, state that the information is not available in the provided text.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        response = model.generate_content(
            f"System Prompt: {system_prompt}\n\nContext:\n---\n{context}\n---\nQuestion: {question}"
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer via Gemini: {e}"

def process_single_question(question: str, text_chunks: List[str], namespace: str) -> str:
    """Process a single question and return the answer"""
    try:
        query_embedding = embedding_model.encode(question).tolist()
        
        # Retrieve from Pinecone
        retrieval_results = index.query(
            vector=query_embedding,
            top_k=20,
            namespace=namespace
        )
        
        if not retrieval_results['matches']:
            return "No relevant information found in the document for this question."

        retrieved_ids = [int(res['id'].split('_')[1]) for res in retrieval_results['matches']]
        initial_chunks = [text_chunks[i] for i in retrieved_ids]

        if not initial_chunks:
            return "No relevant information found in the document for this question."

        # Re-rank
        rerank_pairs = [[question, chunk] for chunk in initial_chunks]
        scores = reranker_model.predict(rerank_pairs)
        
        scored_chunks = list(zip(scores, initial_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        top_chunks = [chunk for score, chunk in scored_chunks[:5]]
        context_str = "\n---\n".join(top_chunks)
        
        # Generate answer
        answer = get_answer_from_llm(question, context_str)
        return answer
        
    except Exception as e:
        return f"Error processing question: {e}"

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, token: str = Depends(verify_token)):
    """Main endpoint that orchestrates the complete, optimized RAG pipeline."""
    print("🚀 Starting optimized submission process...")
    namespace = str(uuid.uuid4())
    print(f"📄 Generated unique namespace: {namespace}")
    
    print("1. Parsing document with unstructured.io...")
    text_chunks = get_text_chunks_unstructured(request.documents)
    print(f"✅ Parsed document into {len(text_chunks)} chunks.")
    
    try:
        print("2. Embedding chunks and upserting to Pinecone...")
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=False).tolist()
        vectors_to_upsert = [(f"vec_{i}", emb) for i, emb in enumerate(embeddings)]
        index.upsert(vectors=vectors_to_upsert, namespace=namespace, batch_size=100)
        print(f"✅ Upserted {len(vectors_to_upsert)} vectors to Pinecone.")

        # OPTIMIZED: Reduced delay to 5 seconds (balance between speed and reliability)
        print("⏳ Waiting for Pinecone to index...")
        time.sleep(5)  # Balance between speed and reliability

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert vectors to Pinecone: {e}")

    print("3. Processing questions in parallel...")
    
    # OPTIMIZED: Process questions in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all questions for parallel processing
        future_to_question = {
            executor.submit(process_single_question, question, text_chunks, namespace): question 
            for question in request.questions
        }
        
        # Collect results in order
        final_answers = []
        for future in future_to_question:
            try:
                answer = future.result(timeout=30)  # 30 second timeout per question
                final_answers.append(answer)
            except Exception as e:
                final_answers.append(f"Error processing question: {e}")
    
    print(f"🎉 Optimized submission process complete. Processed {len(final_answers)} questions.")
    return QueryResponse(answers=final_answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 