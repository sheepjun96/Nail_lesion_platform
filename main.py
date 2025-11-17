from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json, io, base64
from io import BytesIO

from utils.nail_detect import NailDetect
from utils.lesion_predict import LesionPredict

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
    print("Loading models...")
    global nail_detector
    model_path = "models/yolov11-obb.pt"
    nail_detector = NailDetect(model_path)
    print("Nail detection model loaded.")

    global lesion_predictor
    model_path = "models/MedSigLIP"
    lesion_predictor = LesionPredict(model_path, class_names)
    print("Lesion prediction model loaded.")
    print("All models loaded.")

@app.post("/nail_detect/")
async def nail_detect(
    file: UploadFile = File(...),
    is_thumb: bool = Form(False),
    save_dir: str = Form("./static/images")
):
    img_bytes = BytesIO(await file.read())
    results = nail_detector.detect_and_crop(img_bytes, is_thumb=is_thumb, save_dir=save_dir)
    outs = []
    for i, out in enumerate(results):
        outs.append({
            "nail_index": i,
            "obb_info": out["obb_info"]
        })
    return {"results": outs}

@app.post("/plot_nail/")
async def plot_nail(cropped_image: UploadFile = File(...), obb_info: str = Form(...)):
    img_bytes = await cropped_image.read()
    cropped_img = np.array(Image.open(io.BytesIO(img_bytes)))
    
    obb = json.loads(obb_info)
    cx, cy, _, _, _ = obb

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(cropped_img)  
    ax.plot([cx, cx], [0, cropped_img.shape[0]], color='black', linewidth=1)
    ax.plot([0, cropped_img.shape[1]], [cy, cy], color='black', linewidth=1)
    plt.axis('off')

    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    buf.seek(0)
    plt.close(fig)
    img_bytes = buf.getvalue()

    return Response(content=img_bytes, media_type="image/png")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = lesion_predictor.predict(image)
    return result

# uvicorn main:app --reload