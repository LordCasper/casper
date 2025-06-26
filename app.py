import os
import io
import requests
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Google Drive direct download links for your models
CNN_MODEL_URL = "https://drive.google.com/uc?export=download&id=1nO8vvUgAoYGLeQJXacbCl8GhdASs8dQL"
RESNET_MODEL_URL = "https://drive.google.com/uc?export=download&id=1vjrE493mbJV2-DQQ7dDj4NPxpBpv-uHG"

# Local paths to save models
CNN_MODEL_PATH = "cnn_model.pth"
RESNET_MODEL_PATH = "resnet_model.pth"

# Define your CNN and ResNet model architectures here
# (Replace these with your actual model classes)
class YourCNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define your CNN layers here
    
    def forward(self, x):
        # Define forward pass
        pass

class YourResNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define your ResNet layers here
    
    def forward(self, x):
        # Define forward pass
        pass

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading model from {url} ...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {dest_path}")
        else:
            raise RuntimeError(f"Failed to download model: {url}")

def load_models():
    download_file(CNN_MODEL_URL, CNN_MODEL_PATH)
    download_file(RESNET_MODEL_URL, RESNET_MODEL_PATH)
    
    # Load CNN model
    cnn_model = YourCNNModel()
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu')))
    cnn_model.eval()

    # Load ResNet model
    resnet_model = YourResNetModel()
    resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=torch.device('cpu')))
    resnet_model.eval()

    return cnn_model, resnet_model

cnn_model, resnet_model = load_models()

# Image pre-processing function (adjust as needed)
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # or your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # example normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or "model_choice" not in request.form:
        return jsonify({"error": "Missing file or model_choice"}), 400

    file = request.files["file"]
    model_choice = request.form["model_choice"].lower()

    if model_choice not in ["cnn", "resnet", "all"]:
        return jsonify({"error": "Invalid model_choice. Must be 'cnn', 'resnet' or 'all'."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to open image: {e}"}), 400

    input_tensor = preprocess_image(image)

    results = {}

    with torch.no_grad():
        if model_choice in ["cnn", "all"]:
            output_cnn = cnn_model(input_tensor)
            pred_cnn = torch.argmax(output_cnn, dim=1).item()
            results["cnn"] = pred_cnn  # Or map this to label if you have label mapping

        if model_choice in ["resnet", "all"]:
            output_resnet = resnet_model(input_tensor)
            pred_resnet = torch.argmax(output_resnet, dim=1).item()
            results["resnet"] = pred_resnet  # Map to label if needed

    return jsonify(results)

if __name__ == "__main__":
    # Run the Flask app on port 5000 (default)
    app.run(host="0.0.0.0", port=5000)
