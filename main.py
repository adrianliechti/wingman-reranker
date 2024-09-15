import os
import torch
import uvicorn

from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

from transformers import AutoModelForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForSequenceClassification.from_pretrained(
    os.getenv("MODEL", "jinaai/jina-reranker-v2-base-multilingual"),
    torch_dtype="auto",
    trust_remote_code=True,
).to(device)

model.eval()

app = FastAPI(
    title="LLM Platform Reranker"
)

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

    top_n: Optional[int] = None

@app.post("/rerank")
@app.post("/v1/rerank")
async def rerank(request: RerankRequest):
    query = request.query
    documents = request.documents
    
    top_n = request.top_n

    pairs = [[query, doc] for doc in documents]
    scores = model.compute_score(pairs, max_length=1024)

    results = [{"index": i, "document": { "text": doc }, "relevance_score": score} for i, (doc, score) in enumerate(zip(documents, scores))]
    results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    if top_n is not None:
        results = results[:top_n]

    return {
        "model": model.name_or_path,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)