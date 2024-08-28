import torch
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model (replace 'model.pkl' with your actual file)
model = torch.load('fullconnectedNN.pkl', map_location=torch.device('cpu'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        digit = predicted.item()

    # Return the prediction as a JSON response
    return jsonify({'digit': digit})

if __name__ == '__main__':
    app.run(debug=True)
