import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import joblib
from yolo_dataset import YoloDataset  # Import the YOLO dataset loader

# ========================
# 1️⃣ SETUP & CONFIGURATION
# ========================

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Define dataset root
dataset_root = r"C:\Users\avnis\Downloads\X-ray baggage detection.v1-prohibited_items.yolov5pytorch"
import os

dataset_path = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\train"

if os.path.exists(dataset_path):
    print("✅ Path exists:", dataset_path)
    print("📂 Folder contents:", os.listdir(dataset_path))
else:
    print("❌ Path does NOT exist! Check the dataset location.")


# Define dataset paths
train_image_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\train\images"
train_label_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\train\labels"

valid_image_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\valid\images"
valid_label_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\valid\labels"

test_image_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\test\images"
test_label_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\test\labels"

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to ResNet50 input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load datasets using the YOLO dataset loader
train_dataset = YoloDataset(train_image_dir, train_label_dir, transform=transform)
valid_dataset = YoloDataset(valid_image_dir, valid_label_dir, transform=transform)
test_dataset = YoloDataset(test_image_dir, test_label_dir, transform=transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Train Samples: {len(train_dataset)}, Valid Samples: {len(valid_dataset)}, Test Samples: {len(test_dataset)}")

# ==========================
# 2️⃣ LOAD PRETRAINED RESNET50
# ==========================

# Load pre-trained ResNet50 model
resnet = models.resnet50(pretrained=True).to(device)

# Remove the last classification layer to get feature vectors
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()  # Set model to evaluation mode

print("✅ ResNet50 Model Loaded & Modified for Feature Extraction")

# ==========================
# 3️⃣ FUNCTION TO EXTRACT FEATURES
# ==========================

def extract_features(loader, model):
    features = []
    img_names = []

    with torch.no_grad():  # No gradient calculations for efficiency
        for images, img_name in loader:
            images = images.to(device)
            feature = model(images)  # Extract features
            feature = feature.view(feature.size(0), -1)  # Flatten feature maps
            features.append(feature.cpu().numpy())
            img_names.extend(img_name)

    return np.concatenate(features, axis=0), img_names

# ==========================
# 4️⃣ EXTRACT FEATURES FOR DATASET
# ==========================

# Extract features for train, valid, and test sets
print("✅ Extracting Features...")

train_features, train_img_names = extract_features(train_loader, resnet)
valid_features, valid_img_names = extract_features(valid_loader, resnet)
test_features, test_img_names = extract_features(test_loader, resnet)

print("✅ Feature Extraction Completed!")
print(f"Train Features Shape: {train_features.shape}")  # Expected: (num_images, 2048)

# ==========================
# 5️⃣ SAVE FEATURES FOR LATER USE
# ==========================

joblib.dump((train_features, train_img_names), "train_features.pkl")
joblib.dump((valid_features, valid_img_names), "valid_features.pkl")
joblib.dump((test_features, test_img_names), "test_features.pkl")

print("✅ Features Saved Successfully!")

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 6️⃣ LOAD EXTRACTED FEATURES
# ==============================

print("✅ Loading Extracted Features...")

# Load extracted feature vectors
train_features, train_img_names = joblib.load("train_features.pkl")
valid_features, valid_img_names = joblib.load("valid_features.pkl")
test_features, test_img_names = joblib.load("test_features.pkl")

print(f"✅ Train Features Shape: {train_features.shape}")
print(f"✅ Valid Features Shape: {valid_features.shape}")
print(f"✅ Test Features Shape: {test_features.shape}")

# ==============================
# 7️⃣ LOAD LABELS FOR CLASSIFICATION
# ==============================

# Load your corresponding labels (Ensure order matches train_img_names)
# This is a placeholder - You must implement a function to extract labels from your dataset
def load_labels(img_names, label_dict):
    """ Match image names to their labels using label_dict, ensuring filenames are normalized. """
    return np.array([label_dict.get(os.path.splitext(img)[0], -1) for img in img_names])  # -1 for missing labels

# Example: A dictionary mapping image filenames to class labels
label_dict = {
    "image1.jpg": 1,  # Example: '1' for prohibited item, '0' for non-prohibited
    "image2.jpg": 0,
    # Add all images here...
}

# Load labels
train_labels = load_labels(train_img_names, label_dict)
valid_labels = load_labels(valid_img_names, label_dict)
test_labels = load_labels(test_img_names, label_dict)

print(f"✅ Labels Loaded: Train({len(train_labels)}), Valid({len(valid_labels)}), Test({len(test_labels)})")

# ==============================
# 8️⃣ TRAIN A CLASSIFIER
# ==============================

print("🚀 Training Classifier...")

# Initialize Random Forest Classifier
clf = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_split=5, random_state=42)

# Train the classifier using ResNet50 features
clf.fit(train_features, train_labels)

print("✅ Classifier Training Completed!")

# ==============================
# 9️⃣ VALIDATE MODEL PERFORMANCE
# ==============================

print("📊 Validating Model...")

# Predict on validation set
valid_preds = clf.predict(valid_features)

# Compute accuracy
valid_accuracy = accuracy_score(valid_labels, valid_preds)
print(f"✅ Validation Accuracy: {valid_accuracy:.4f}")

# Generate a classification report
print("📊 Classification Report:")
print(classification_report(valid_labels, valid_preds))

# ==============================
# 🔟 TEST MODEL ON UNSEEN DATA
# ==============================

print("🛠 Testing on Unseen Data...")

# Predict on test set
test_preds = clf.predict(test_features)

# Compute test accuracy
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# Generate a classification report for test data
print("📊 Test Set Performance:")
print(classification_report(test_labels, test_preds))

# ==============================
# 1️⃣1️⃣ SAVE TRAINED CLASSIFIER
# ==============================

joblib.dump(clf, "prohibited_items_classifier.pkl")
print("✅ Model Saved Successfully! 🚀")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd

# Load trained classifier
clf = joblib.load("prohibited_items_classifier.pkl")

# Load extracted features and labels
train_features, train_img_names = joblib.load("train_features.pkl")
valid_features, valid_img_names = joblib.load("valid_features.pkl")
test_features, test_img_names = joblib.load("test_features.pkl")

train_labels = load_labels(train_img_names, label_dict)
valid_labels = load_labels(valid_img_names, label_dict)
test_labels = load_labels(test_img_names, label_dict)

# Predictions
train_preds = clf.predict(train_features)
valid_preds = clf.predict(valid_features)
test_preds = clf.predict(test_features)

# Compute accuracy
train_accuracy = accuracy_score(train_labels, train_preds)
valid_accuracy = accuracy_score(valid_labels, valid_preds)
test_accuracy = accuracy_score(test_labels, test_preds)

# ==============================
# 📊 BAR PLOT FOR ACCURACY
# ==============================
plt.figure(figsize=(8, 5))
plt.bar(['Train', 'Validation', 'Test'], [train_accuracy, valid_accuracy, test_accuracy], color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy on Train, Validation, and Test Sets')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ==============================
# 📉 CONFUSION MATRIX
# ==============================
test_cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Prohibited', 'Prohibited'], yticklabels=['Non-Prohibited', 'Prohibited'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.show()

# ==============================
# 🔥 CLASSIFICATION REPORT HEATMAP
# ==============================
report = classification_report(test_labels, test_preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Classification Report")
plt.show()