import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification

class LesionPredict:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path, num_labels=len(class_names)
        ).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def predict(self, pil_image):
        image = pil_image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probabilities, dim=1)
            predicted_class = self.class_names[top_idx.item()]
            return {
                "predicted_class": predicted_class,
                "probability": float(top_prob.cpu().item())
            }