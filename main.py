from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from utils.nail_detect import NailDetect
from utils.lesion_predict import LesionPredict

from io import BytesIO

app = FastAPI()

class_names = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "Onychomycosis",
    "blue_finger",
    "clubbing",
    "pitting",
    "psoriasis"
]

@app.on_event("startup")
def load_model():
    global lesion_predictor
    model_path = "models/MedSigLIP"
    lesion_predictor = LesionPredict(model_path, class_names)

    global nail_detector
    model_path = "models/yolov11-obb.pt"
    nail_detector = NailDetect(model_path)

@app.post("/nail_detect/")
async def nail_detect(file: UploadFile = File(...)):
    img_bytes = BytesIO(await file.read())
    results = nail_detector.detect_and_crop(img_bytes)
    outs = []
    for i, out in enumerate(results):
        outs.append({
            "nail_index": i,
            "obb_info": out["obb_info"]
        })
    return {"results": outs}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = lesion_predictor.predict(image)
    return result
