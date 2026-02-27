from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import google.generativeai as genai
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class Comment(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

@app.post("/comment")
def analyze_comment(body: Comment):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=SentimentResponse,
            )
        )
        prompt = f"""Analyze the sentiment of this comment and return JSON with:
- sentiment: exactly 'positive', 'negative', or 'neutral'
- rating: integer 1-5 (5=very positive, 1=very negative, 3=neutral)

Comment: {body.comment}"""
        
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return {"sentiment": result["sentiment"], "rating": result["rating"]}
    except Exception as e:
        return {"sentiment": "neutral", "rating": 3}
