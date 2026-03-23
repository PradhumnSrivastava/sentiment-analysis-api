from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

def predict_sentiment(text):
    result = classifier(text)[0]
    return result["label"], result["score"]