from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_sentiment

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: InputText):
    label, score = predict_sentiment(data.text)
    return {
        "sentiment": label,
        "confidence": float(score)
    }