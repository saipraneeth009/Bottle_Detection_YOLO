import os
import cv2
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Function to load images
def load_images(image_folder):
    images = []
    image_names = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):  # Assuming images are in .jpg format
            img = cv2.imread(os.path.join(image_folder, filename))
            if img is not None:
                images.append(img)
                image_names.append(filename)
    return images, image_names

# Function to load labels from txt files
def load_labels(label_folder, image_names):
    labels = {}
    for image_name in image_names:
        label_file = os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt')
        with open(label_file, 'r') as file:
            labels[image_name] = [line.strip().split() for line in file]
    return labels

# Function to draw bounding boxes on the image
def draw_bboxes(image, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()  # get box coordinates in (top, left, bottom, right) format
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            label = f"Class: {class_id}, Conf: {conf:.2f}"
            cv2.putText(image, label, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Load images and their corresponding labels
image_folder = 'test/images'
label_folder = 'test/labels'
save_folder = 'predicted_images'  # Folder to save images with bounding boxes

# Create save folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

images, image_names = load_images(image_folder)
labels = load_labels(label_folder, image_names)

# Run inference and collect predictions
y_true = []
y_pred = []
prediction_file = open("predictions.txt", "w")

for image, image_name in zip(images, image_names):
    # Run inference
    results = model(image)
    
    # Process each detection in the results
    predicted_classes = []
    predicted_confidences = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            predicted_classes.append(int(box.cls[0].item()))
            predicted_confidences.append(box.conf[0].item())
    
    # Get the true classes from labels
    true_classes = [int(label[0]) for label in labels[image_name]]
    
    # Draw bounding boxes on the image
    image_with_bboxes = draw_bboxes(image.copy(), results)
    
    # Save the image with bounding boxes
    save_path = os.path.join(save_folder, image_name)
    cv2.imwrite(save_path, image_with_bboxes)
    
    # Write the predictions to the file
    for class_id, confidence in zip(predicted_classes, predicted_confidences):
        prediction_file.write(f"{image_name} {class_id} {confidence:.2f}\n")
    
    y_true.extend(true_classes)
    y_pred.extend(predicted_classes)

# Close the prediction file
prediction_file.close()

# Generate confusion matrix and classification report
class_names = ['glass-bottle', 'plastic-bottle']  # Adjust based on your class names
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=class_names)

# Print the results
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)