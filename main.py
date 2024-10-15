from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import tensorflow as tf
import os
from pathlib import Path
# Initialize Flask app
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the model and scaler
model = tf.keras.models.load_model(f'{BASE_DIR}/emotion_detection_model_more_layers_v3.h5')
scaler = joblib.load(f'{BASE_DIR}/scaler_more_layers_v3.save')

# Label mapping
label_mapping = {0: 'angry', 1: 'happy', 2: 'more_angry', 3: 'neutral', 4: 'sad', 5: 'slightly_angry'}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Function to extract features from an audio file
def extract_features_from_audio(audio_path):
    # Load audio file (assuming the file is max 10 seconds long)
    y, sr = librosa.load(audio_path, duration=10.0)  # Load up to 10 seconds
    # y = noise_reduction(y)
    # y = apply_agc(y, ref_db=-20)
    # Extract features (energy, zcr, rms, pitch, jitter, shimmer, mfcc, chroma, mel)
    energy = np.sum(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) 
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    # Pitch (F0)
   
    # Jitter and Shimmer (approximated by variations in pitch and energy)
    jitter = np.std(pitches[pitches > 0]) / pitch_mean
    shimmer = np.std(librosa.feature.rms(y=y)) / rms    
    # Combine all features into a single array
    features = np.hstack((energy, zcr, rms, pitch_mean, jitter, shimmer, mfccs, chroma, mel))
    
    return features

def predicting_emotion(audio_path, model, scaler):
    # Extract features
    features = extract_features_from_audio(audio_path)

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict using the model
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction)
    y = ['angry','happy','more_angry','neutral','sad','slightly_angry']
    label_mapping = {label: i for i, label in enumerate(np.unique(y))}

    # Map numerical labels back to original labels
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = inverse_label_mapping[predicted_class]
    
    return predicted_label

def analyze_anger(text):
    
    # Determine the level of anger based on keyword counts
    if text == "angry":
        return "Angry"
    elif text == "more_angry":
        return "Very Angry"
    elif text == "slightly_angry":
        return "Slightly Angry"
    else:
        return "Not Angry"
# Define the prediction route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_emotion():
    # Check if an audio file is sent in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Get the file from the request
    file = request.files['file']
    
    # Save the file to a temporary location
    file_path = os.path.join(f'{BASE_DIR}/audios/', file.filename)
    file.save(file_path)

    # # Make prediction using the model
    response = analyze_anger(predicting_emotion(file_path, model, scaler))    
    # Remove the temporary audio file
    os.remove(file_path)
    
    # Return the predicted emotion as JSON
    return jsonify({'emotion': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
