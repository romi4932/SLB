import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os

# Determine number of classes and create label maps
training_folder = r'C:\Users\romai\Downloads\7223446\OpenEarthMap\OpenEarthMap_wo_xBD\organized_data\train'
filenames = os.listdir(training_folder)
labels = [filename.split('_')[0] for filename in filenames]
unique_labels = set(labels)
num_classes = len(unique_labels)

label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {v: k for k, v in label_map.items()}

# Load the trained VGG model
model_path = r'C:\Users\romai\Downloads\7223446\OpenEarthMap\OpenEarthMap_wo_xBD\trained_model_vgg16.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=False)
vgg16.classifier[6] = torch.nn.Linear(vgg16.classifier[6].in_features, num_classes)
vgg16.load_state_dict(torch.load(model_path, map_location=device))
vgg16.to(device)
vgg16.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title('Area Name Label Predictor')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Predict the label
    with torch.no_grad():
        outputs = vgg16(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()

        st.write(f'Predicted Index: {predicted_index}')
        if predicted_index in reverse_label_map:
            label = reverse_label_map[predicted_index]
            st.write(f'Predicted Label: {label}')
        else:
            st.write("Error: Predicted index not found in label map.")
