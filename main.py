from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("AIPROXY_TOKEN"),
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

class Comment(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment")
def analyze_comment(body: Comment):
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze the sentiment of the comment. Return sentiment as exactly 'positive', 'negative', or 'neutral'. Return rating as integer 1-5 (5=most positive, 1=most negative)."},
                {"role": "user", "content": body.comment}
            ],
            response_format=SentimentResponse,
        )
        result = response.choices[0].message.parsed
        return {"sentiment": result.sentiment, "rating": result.rating}
    except Exception as e:
        return {"sentiment": "neutral", "rating": 3}
