import torch
import torchvision.transforms as transforms
import sys
from config import input_size
from torchvision.io import read_image

# Check the number of command-line arguments
if len(sys.argv) < 2:
    print("Usage: python predict_label.py <image_path>")
    sys.exit(1)

# Load the trained model
model_path = './model.pth'
model = torch.load(model_path)

# Define the image transformation
transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])

# Load and preprocess the image
image_path = sys.argv[1]

image = read_image(image_path)
image = transform(image)
image = image.reshape(-1, input_size)

# Perform inference
with torch.no_grad():
    model.eval()
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Print the predicted label
label = predicted.item()
pred = None

if label == 0:
    pred = 'Arborio'

if label == 1:
    pred = 'Basmati'

if label == 2:
    pred = 'Ipsala'

if label == 3:
    pred = 'Jasmine'

if label == 4:
    pred = 'Karacadag'

print(pred)