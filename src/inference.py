import torch
from flask import Flask, request, jsonify
import torchvision.transforms as transforms
from PIL import Image
from model import CNN
import io

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('/app/models/cat_dog_cnn.pth', map_location=device))
model.to(device)
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    input_tensor = transform_image(img_bytes)
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    
    label = "Cat" if prediction == 0 else "Dog"
    return jsonify({'prediction': label})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
