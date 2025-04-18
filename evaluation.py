import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from feature_extraction import build_feature_extractor


# Load the feature extraction model
feature_extractor = build_feature_extractor()


def extract_features(img):
    img = cv2.resize(img, (128, 128))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img)
    features = features.reshape(features.shape[0], -1)
    print(features.shape)
    return features


# Function to calculate cosine similarity
def evaluate_signature(image, username):
    user_folder = os.path.join("features", username)

    if not os.path.exists(user_folder):
        print(f"No features found for user: {username}")
        return 0.0

    features_path = os.path.join(user_folder, "signature_features.npy")

    user_features = np.load(features_path)
    if user_features.size == 0:
        print(f"No feature vectors available for user: {username}")
        return 0.0

    image_features = extract_features(image)

    similarities = cosine_similarity(image_features, user_features)

    return similarities[0][0]
