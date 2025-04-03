
from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

class SimpleNLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

nlp_model = SimpleNLPModel()
nlp_model.load_state_dict(torch.load("nlp_model_pytorch/nlp_model.pt", map_location=torch.device('cpu')))
nlp_model.eval()

image_model = tf.keras.models.load_model("image_model_tf/image_model.h5")

def predict_text():
    vec = torch.rand(1, 100)
    with torch.no_grad():
        out = nlp_model(vec)
        pred = torch.argmax(out).item()
    return f"Predicted class from text: {pred}"

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((64, 64))
    arr = np.array(image) / 255.0
    arr = arr.reshape(1, 64, 64, 3)
    pred = np.argmax(image_model.predict(arr))
    return f"Predicted class from image: {pred}"

@app.get("/")
def home():
    return {"message": "AI Assistant ready!"}

@app.get("/predict-text")
def text_route():
    return {"result": predict_text()}

@app.post("/predict-image")
def image_route(file: UploadFile = File(...)):
    img_bytes = file.file.read()
    return {"result": predict_image(img_bytes)}
