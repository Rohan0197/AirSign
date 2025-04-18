import cv2
import os
import glob
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Build feature extraction model
def build_feature_extractor():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)

    return model


# Load data for feature extraction
def load_data(username):
    x, y = [], []

    user_path = os.path.join("signatures", username)

    if not os.path.isdir(user_path):
        raise ValueError(f"User {username} not found")

    for img_path in glob.glob(os.path.join(user_path, "signature_*.png")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = preprocess_input(img)
        x.append(img)
        y.append(0)

    x = np.array(x)
    y = np.array(y)

    return x, y


# Extract features using the VGG16 model
def extract_features(model, x):
    features = model.predict(x)
    # Flatten the features into a 1D vector per image
    features = features.reshape(features.shape[0], -1)
    return features


# Save Features
def save_features(username):
    x, y = load_data(username)

    feature_extractor = build_feature_extractor()

    features = extract_features(feature_extractor, x)
    combined_feature = np.mean(features, axis=0).reshape(1, -1)
    # Create the features directory if it doesn't exist
    features_folder = "features"
    os.makedirs(features_folder, exist_ok=True)

    # Create a subdirectory for the user
    user_folder = os.path.join(features_folder, username)
    os.makedirs(user_folder, exist_ok=True)

    # Save the features and labels in the user-specific folder
    np.save(os.path.join(user_folder, "signature_features.npy"), combined_feature)

    print(f"Extracted features shape: {combined_feature.shape}")
