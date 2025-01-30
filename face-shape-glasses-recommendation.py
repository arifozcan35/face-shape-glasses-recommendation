# Cell
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import torchvision
import torch.nn as nn

class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Load Model
# openCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = torchvision.models.efficientnet_b4(pretrained=True)
num_classes = 5
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

# EfficientNet-B4
model.load_state_dict(torch.load('face_shape_model.pth', map_location=torch.device('cpu')))

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Face shape to glasses mapping
glasses_recommendations = {
    "Heart": "Cat-eye glasses, Round frames",
    "Oblong": "Wide frames, oversized glasses",
    "Oval": "Any frame type, Square or Rectangular frames",
    "Round": "Square frames, Angular glasses",
    "Square": "Round frames, Oval glasses"
}

# GUI Functions
def upload_image():
    global img_path, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if img_path:
        image = Image.open(img_path)
        image = image.resize((300, 300))
        img_display = ImageTk.PhotoImage(image)
        canvas.create_image(150, 150, image=img_display)

def predict_image():
    if not img_path:
        result_label.config(text="Please upload a picture first!")
        return
    
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = class_labels[predicted.item()]
    
    result_label.config(text=f"Predicted Face Shape: {prediction}")
    glasses_recommendation = glasses_recommendations.get(prediction, "No recommendation available")
    glasses_label.config(text=f"Recommended Glasses: {glasses_recommendation}")

# GUI Setup
root = tk.Tk()
root.title("Face Shape Classifier")
root.geometry("600x650")
root.configure(bg="black")

title_label = Label(root, text="Face Shape Classifier", font=("Arial", 18, "bold"), bg="black", fg="white")
title_label.pack(pady=10)

canvas = Canvas(root, width=300, height=300, bg="gray", highlightbackground="white")
canvas.pack(pady=10)
canvas.create_text(150, 150, text="Upload Image", fill="white", font=("Arial", 10))

upload_btn = Button(root, text="Upload", command=upload_image, bg="black", fg="white")
upload_btn.pack(pady=5)

predict_btn = Button(root, text="Predict", command=predict_image, bg="black", fg="white")
predict_btn.pack(pady=5)

result_label = Label(root, text="", font=("Arial", 12), bg="black", fg="white")
result_label.pack(pady=20)

glasses_label = Label(root, text="", font=("Arial", 12), bg="black", fg="white")
glasses_label.pack(pady=10)

img_path = ""
img_display = None

root.mainloop()


# python face-shape-glasses-recommendation.py


