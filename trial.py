import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import joblib
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
data_dir = "dataset"
train_img_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\train\images"
train_label_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\train\labels"
valid_img_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\valid\images"
valid_label_dir =r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\valid\labels"
test_img_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\test\images"
test_label_dir = r"C:\Users\avnis\PycharmProjects\pythonProject1\Minor Project\X-ray baggage detection.v1-prohibited_items.yolov5pytorch\test\labels"


# Custom dataset class
class YoloDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# DataLoader
train_dataset = YoloDataset(train_img_dir, transform)
valid_dataset = YoloDataset(valid_img_dir, transform)
test_dataset = YoloDataset(test_img_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
model = model.to(device)
model.eval()


def extract_features(data_loader, feature_file):
    features, img_names = [], []
    with torch.no_grad():
        for images, names in data_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            img_names.extend(names)
    features = np.concatenate(features, axis=0)
    joblib.dump((features, img_names), feature_file)


if not os.path.exists("train_features.pkl"):
    extract_features(train_loader, "train_features.pkl")
if not os.path.exists("valid_features.pkl"):
    extract_features(valid_loader, "valid_features.pkl")
if not os.path.exists("test_features.pkl"):
    extract_features(test_loader, "test_features.pkl")

# Load features
train_features, train_img_names = joblib.load("train_features.pkl")
valid_features, valid_img_names = joblib.load("valid_features.pkl")
test_features, test_img_names = joblib.load("test_features.pkl")


def load_labels(img_names, label_dir):
    labels = []
    for img_name in img_names:
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels.append(int(f.readline().split()[0]))
        else:
            labels.append(-1)
    return np.array(labels)


train_labels = load_labels(train_img_names, train_label_dir)
valid_labels = load_labels(valid_img_names, valid_label_dir)
test_labels = load_labels(test_img_names, test_label_dir)

# Train Random Forest with Hyperparameter Tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
clf.fit(train_features, train_labels)

# Save best model
joblib.dump(clf.best_estimator_, "rf_classifier.pkl")

# Load classifier and evaluate
classifier = joblib.load("rf_classifier.pkl")
valid_preds = classifier.predict(valid_features)
test_preds = classifier.predict(test_features)

valid_acc = accuracy_score(valid_labels, valid_preds)
test_acc = accuracy_score(test_labels, test_preds)

print(f"Validation Accuracy: {valid_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Classification Report
print("Validation Report:\n", classification_report(valid_labels, valid_preds))
print("Test Report:\n", classification_report(test_labels, test_preds))

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(["Validation", "Test"], [valid_acc, test_acc], color=['blue', 'green'])
plt.ylabel("Accuracy")
plt.title("Model Performance")
plt.show()

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Set Confusion Matrix")
plt.show()
