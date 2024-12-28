import os
import uvicorn

from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

from sentence_transformers import CrossEncoder

model_name = os.getenv("MODEL", "jinaai/jina-reranker-v2-base-multilingual")
model = CrossEncoder(model_name, trust_remote_code=True)

app = FastAPI(
    title="LLM Platform Reranker"
)

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

    top_n: Optional[int] = None

@app.post("/rerank")
@app.post("/v1/rerank")
def rerank(request: RerankRequest):
    query = request.query
    documents = request.documents
    
    top_n = request.top_n

    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)

    results = [
        {
            "index": i,
            "document": {"text": doc},
            "relevance_score": float(scores[i])
        }
        for i, doc in enumerate(documents)
    ]
    
    results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    if top_n is not None:
        results = results[:top_n]

    return {
        "model": model_name,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)