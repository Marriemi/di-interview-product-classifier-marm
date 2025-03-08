import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import json


MODEL_PATH = "app/bert_model.pth" 
LABELS_PATH = "app/labels.json" 
NUM_LABELS = 383  

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
model.to(device)
model.eval()


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)
    
app = FastAPI(title="Product Classification API", version="1.0")


class Item(BaseModel):
    title: str

@app.post("/classify")
def classify(item: Item):
    try:
        
        inputs = tokenizer(item.title, return_tensors="pt", truncation=True, padding=True, max_length=32)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0] 

    
        top_3_indices = np.argsort(probs)[-3:][::-1]  
        top_3_results = [{"product_type": class_names[str(idx)], "score": float(probs[idx])} for idx in top_3_indices]

 
        predicted_class = class_names[str(top_3_indices[0])]

       
        return {
            "title": item.title,
            "top_3_results": top_3_results,
            "product_type": predicted_class
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
