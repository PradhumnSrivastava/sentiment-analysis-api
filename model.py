from transformers import pipeline

classifier = None

def predict_sentiment(text):
    global classifier

    if classifier is None:
        classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=-1
        )

    result = classifier(text)[0]
    return result["label"], result["score"]