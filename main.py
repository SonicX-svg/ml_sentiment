from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()
classifier = pipeline("sentiment-analysis")
class Item(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
