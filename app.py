from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import joblib
import tensorflow as tf
import os
from pathlib import Path
import logging

# Initialize FastAPI app
app = FastAPI()
BASE_DIR = Path(__file__).resolve(strict=True).parent
model = None
scaler = None
# Load the model and scaler
try:
    model = tf.keras.models.load_model(f'{BASE_DIR}/emotion_detection_model_more_layers_v3.h5')
    scaler = joblib.load(f'{BASE_DIR}/scaler_more_layers_v3.save')
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
# Label mapping
# label_mapping = {0: 'not_angry', 1: 'slightly_angry', 2: 'angry', 3: 'very_angry'}
label_mapping = {0: 'angry', 1: 'happy', 2: 'more_angry', 3: 'neutral', 4: 'sad', 5: 'slightly_angry'}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Function to extract features from an audio file
def extract_features_from_audio(audio_path):
    try:
        # Load audio file (assuming the file is max 10 seconds long)
        y, sr = librosa.load(audio_path, duration=10.0)  # Load up to 10 seconds
        # Extract features (energy, zcr, rms, pitch, jitter, shimmer, mfcc, chroma, mel)
        energy = np.sum(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        rms = np.mean(librosa.feature.rms(y=y))
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0])
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)

        # Jitter and Shimmer (approximated by variations in pitch and energy)
        jitter = np.std(pitches[pitches > 0]) / pitch_mean
        shimmer = np.std(librosa.feature.rms(y=y)) / rms    
        # Combine all features into a single array
        features = np.hstack((energy, zcr, rms, pitch_mean, jitter, shimmer, mfccs, chroma, mel))
    
        return features
    except Exception as e:
        logging.error(f"Error extracting features from audio: {e}")
        return None    

def predicting_emotion(audio_path):
    try:

        # Extract features
        features = extract_features_from_audio(audio_path)

        # Scale the features
        features_scaled = scaler.transform([features])

        # Predict using the model
        prediction = model.predict(features_scaled)
        predicted_class = np.argmax(prediction)
        print(prediction)
        # Map numerical labels back to original labels
        predicted_label = label_mapping[predicted_class]
        
        return predicted_label
    except Exception as e:
        logging.error(f"Error in predicting emotion: {e}")
        return logging.error(f"Error in predicting emotion: {e}")
    
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
@app.get('/health')
def health_check():
    return JSONResponse(content={'emotion': 'It is working fine'})

# Define the prediction route
@app.post('/predict')
async def predict_emotion(file: UploadFile = File(...)):
    try:

        # Save the file to a temporary location
        file_path = os.path.join(f'{BASE_DIR}/audios/', file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Make prediction using the model
        response = analyze_anger(predicting_emotion(file_path))    
        # Remove the temporary audio file
        os.remove(file_path)
        
        # Return the predicted emotion as JSON
        return JSONResponse(content={'emotion': response})
    except Exception as e:
        logging.error(f"Error in /predict route: {e}")
        return JSONResponse(content={'error': str(e)})
    
# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
