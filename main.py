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
                temperature=0.1,
            )
        )
        prompt = f"""You are a sentiment analysis expert. Analyze the sentiment of the following comment and classify it.

Rules:
- sentiment must be EXACTLY one of: 'positive', 'negative', or 'neutral'
- 'positive': comment expresses satisfaction, happiness, praise, or excitement
- 'negative': comment expresses dissatisfaction, disappointment, criticism, or anger
- 'neutral': comment is neither clearly positive nor negative
- rating is an integer from 1 to 5:
  * 5 = very positive (enthusiastic praise)
  * 4 = positive (satisfied/pleased)
  * 3 = neutral (neither good nor bad)
  * 2 = negative (disappointed/dissatisfied)
  * 1 = very negative (angry/strongly critical)

Comment: {body.comment}

Return JSON with 'sentiment' and 'rating' keys."""

        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return {"sentiment": result["sentiment"], "rating": result["rating"]}
    except Exception as e:
        return {"error": str(e)}
