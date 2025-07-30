import os
import uuid
import requests
import tempfile
import time
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
    raise ValueError("One or more required environment variables are not set.")

# Initialize services that have a low memory footprint
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# OPTIMIZED: Models are now loaded inside the processing function to save memory at startup.
embedding_model = None
reranker_model = None

app = FastAPI(title="High-Accuracy Intelligent Query-Retrieval System")
security = HTTPBearer()

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str = Field(..., example="URL to the policy PDF, DOCX, or EML")
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Authentication ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# --- Core Logic Functions ---
def get_text_chunks_unstructured(doc_url: str) -> List[str]:
    """Downloads and parses a document using unstructured.io."""
    temp_file_path = None
    try:
        response = requests.get(doc_url, timeout=20)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(response.content)
            temp_file_path = tmp.name
        
        elements = partition(filename=temp_file_path, strategy="fast")
        chunks = chunk_by_title(elements, max_characters=1000)
        chunk_texts = [chunk.text for chunk in chunks]

        if not chunk_texts:
            raise ValueError("Document could not be parsed into meaningful chunks.")
        return chunk_texts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_answer_from_llm(question: str, context: str) -> str:
    """Generates an answer from the Gemini model."""
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
    """Processes a single question from parsing to generation."""
    global embedding_model, reranker_model
    try:
        # OPTIMIZED: Lazy load models only when the first question is processed.
        if embedding_model is None:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        if reranker_model is None:
            reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        query_embedding = embedding_model.encode(question).tolist()
        
        retrieval_results = index.query(vector=query_embedding, top_k=20, namespace=namespace)
        
        if not retrieval_results['matches']:
            return "No relevant information found in the document for this question."

        retrieved_ids = [int(res['id'].split('_')[1]) for res in retrieval_results['matches']]
        initial_chunks = [text_chunks[i] for i in retrieved_ids]

        if not initial_chunks:
            return "No relevant information found in the document for this question."

        rerank_pairs = [[question, chunk] for chunk in initial_chunks]
        scores = reranker_model.predict(rerank_pairs)
        
        scored_chunks = list(zip(scores, initial_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        top_chunks = [chunk for score, chunk in scored_chunks[:5]]
        context_str = "\n---\n".join(top_chunks)
        
        answer = get_answer_from_llm(question, context_str)
        return answer
    except Exception as e:
        return f"Error processing question: {e}"

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, token: str = Depends(verify_token)):
    """Main endpoint that orchestrates the complete, optimized RAG pipeline."""
    namespace = str(uuid.uuid4())
    
    text_chunks = get_text_chunks_unstructured(request.documents)
    
    try:
        # Temporarily load the embedding model just for this step
        temp_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = temp_embedding_model.encode(text_chunks, show_progress_bar=False).tolist()
        del temp_embedding_model # Free up memory immediately

        vectors_to_upsert = [(f"vec_{i}", emb) for i, emb in enumerate(embeddings)]
        index.upsert(vectors=vectors_to_upsert, namespace=namespace, batch_size=100)
        
        time.sleep(2)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert vectors to Pinecone: {e}")

    # OPTIMIZED: Reduced max_workers to manage memory during parallel processing.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_question = {
            executor.submit(process_single_question, question, text_chunks, namespace): question 
            for question in request.questions
        }
        
        question_to_answer = {}
        for future in future_to_question:
            question = future_to_question[future]
            try:
                answer = future.result(timeout=25)
                question_to_answer[question] = answer
            except Exception as e:
                question_to_answer[question] = f"Error processing question: {e}"
    
    final_answers = [question_to_answer[q] for q in request.questions]
    
    return QueryResponse(answers=final_answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

