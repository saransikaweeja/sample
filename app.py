import torch
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import io
import traceback

app = Flask(__name__)

# Define your model class (if loading state_dict)
class FullyConnectedNN(torch.nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        # Define your layers here
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the model
def load_model():
    model = FullyConnectedNN()
    try:
        # Try to load the model's state_dict
        state_dict = torch.load('fullyconnectedNN.pth', map_location=torch.device('cpu'))
        print("State dict keys:", state_dict.keys())  # Print state dict keys for debugging

        # Load the state_dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        model.eval()
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        model = None
    return model

# Initialize the model
model = load_model()

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((28, 28)),                  
    transforms.ToTensor(),                       
    transforms.Normalize((0.1307,), (0.3081,))   
])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load.'}), 500

    # Get the file from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
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
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())  # Print stack trace for debugging
        return jsonify({'error': f"Failed to process the image: {e}"}), 500


# Test route to check model loading
@app.route('/test_model', methods=['GET'])
def test_model():
    try:
        if model is None:
            return "Model failed to load.", 500
        else:
            return "Model loaded successfully.", 200
    except Exception as e:
        print(f"Error during model test: {e}")
        print(traceback.format_exc())
        return f"Error during model test: {e}", 500
    
if __name__ == '__main__':
    app.run(debug=True)
