from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io

app = Flask(__name__)

# Load both models once at startup
cnn_model = torch.load('cnn_model.pth', map_location='cpu')
cnn_model.eval()

resnet_model = torch.load('resnet_model.pth', map_location='cpu')
resnet_model.eval()

# Label mapping (adjust this to your 32 classes)
label_map = ['alef', 'baa', 'taa', 'thaa', 'jeem', 'haa', 'khaa', 'dal', 'thal',
             'raa', 'zay', 'seen', 'sheen', 'saad', 'daad', 'taa2', 'zaa', 'ain',
             'ghain', 'faa', 'qaaf', 'kaaf', 'laam', 'meem', 'noon', 'haa2',
             'waw', 'yaa', 'hamza', 'lam-alef', 'laam2', 'al']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust if needed for ResNet
    transforms.ToTensor(),
])

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        _, predicted = torch.max(probs, 1)
    return label_map[predicted.item()]

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'Missing image or model field'}), 400

    file = request.files['file']
    model_choice = request.form['model'].lower()

    try:
        image = Image.open(file).convert('RGB')
        tensor = transform(image).unsqueeze(0)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    result = {}

    if model_choice == 'cnn':
        result['cnn'] = predict(cnn_model, tensor)
    elif model_choice == 'resnet':
        result['resnet'] = predict(resnet_model, tensor)
    elif model_choice == 'all':
        result['cnn'] = predict(cnn_model, tensor)
        result['resnet'] = predict(resnet_model, tensor)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
