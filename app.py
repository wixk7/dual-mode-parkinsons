import os
import numpy as np
import torchaudio
import torch
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import sounddevice as sd
import wavio
import librosa
import warnings
import serial  # Import pyserial for serial communication
import time

# Initialize Serial Communication
arduino_port = '/dev/tty.usbmodem1201'  # Update with your Arduino's serial port
baud_rate = 9600  # Set the same baud rate as used in the Arduino code
try:
    arduino = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)  # Wait for the serial connection to initialize
    print(f"Connected to Arduino on port {arduino_port}")
except serial.SerialException as e:
    print(f"Could not open port {arduino_port}: {e}")

# Suppress Torchaudio's UserWarning (optional)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Global variables for normalization and threshold
feature_min = None
feature_max = None
weights = None  # Will define later
SEVERITY_THRESHOLD = 2.5  # Set the global threshold for severity score here

# Function to extract features from audio
def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path, backend="soundfile")  # Use backend keyword to avoid warning
    waveform = waveform.numpy().squeeze()
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=6)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    zcr_mean = np.mean(zcr)
    
    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    spec_centroid_mean = np.mean(spec_centroid)
    
    # Jitter (approximate)
    pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
    pitches = pitches[pitches > 0]
    jitter = np.abs(np.diff(pitches)).mean() / pitches.mean() if len(pitches) > 1 else 0.0
    
    # Shimmer (approximate)
    frame_amplitudes = librosa.feature.rms(y=waveform).squeeze()
    shimmer = np.abs(np.diff(frame_amplitudes)).mean() / frame_amplitudes.mean() if len(frame_amplitudes) > 1 else 0.0
    
    # Harmonic-to-noise ratio (HNR)
    try:
        _, hnr = librosa.effects.hpss(y=waveform)
        hnr_value = np.mean(hnr)
    except:
        hnr_value = 0.0
    
    features = np.concatenate((
        mfcc_mean,
        np.array([zcr_mean, spec_centroid_mean, jitter, shimmer, hnr_value])
    ))
    return features

# Dataset preparation
def load_dataset(folder):
    data, labels = [], []
    class_labels = {'HC_AH': 0, 'PD_AH': 1}
    for class_dir in os.listdir(folder):
        class_path = os.path.join(folder, class_dir)
        if not os.path.isdir(class_path):
            continue
        label = class_labels.get(class_dir)
        if label is None:
            continue
        for filename in os.listdir(class_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(class_path, filename)
                features = extract_features(file_path)
                data.append(features)
                labels.append(label)
    return np.array(data), np.array(labels)

# Load the dataset
dataset_path = 'dataset'  # Update with your dataset path
data, labels = load_dataset(dataset_path)

# Normalize features for severity estimation
feature_min = np.min(data, axis=0)
feature_max = np.max(data, axis=0)
feature_range = feature_max - feature_min
feature_range[feature_range == 0] = 1

# Define weights for severity estimation
weights = np.array([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # MFCC coefficients weights
    0.1,  # Zero-crossing rate
    0.1,  # Spectral centroid
    0.2,  # Jitter
    0.2,  # Shimmer
    0.15  # HNR
])
weights = weights / weights.sum()

# Scale the features for classification
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Train the classification model
clf_model = RandomForestClassifier(n_estimators=100)
clf_model.fit(X_scaled, labels)

# Streamlit UI
st.title("Parkinson's Audio Classification App")
st.write("Upload an audio file or record live audio to classify it as Healthy or Parkinson Diagnosed.")

# Estimate severity score function
def estimate_severity(features):
    normalized_features = (features - feature_min) / feature_range
    if len(weights) != len(normalized_features):
        st.error("Mismatch in the number of features and weights.")
        return 0.0
    severity_score = np.dot(normalized_features, weights) * 5 + 1
    return severity_score

# Classify recording based on the global severity threshold
def classify_audio_from_recording(features):
    severity_score = estimate_severity(features)
    return ("Healthy", 0.0) if severity_score < SEVERITY_THRESHOLD else ("Parkinson Diagnosed", severity_score)

def send_severity_to_arduino(severity_score):
    score_int = min(9, max(0, int(severity_score)))
    message = f"{score_int}"
    print(f"Sending severity score to Arduino: {message}")
    arduino.write(message.encode())
    time.sleep(0.5)

# Recording feature handling
if st.button("Record"):
    fs = 16000
    duration = 5
    st.write("Recording for 5 seconds...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wavio.write("recorded_audio.wav", recording, fs, sampwidth=2)
        st.write("Recording complete.")
        st.audio("recorded_audio.wav", format="audio/wav")
        
        features = extract_features("recorded_audio.wav")
        result, severity_score = classify_audio_from_recording(features)
        st.write(f"The audio has been classified as: **{result}**")
        
        if result == "Parkinson Diagnosed":
            st.write(f"Estimated Severity Score: **{severity_score:.2f}**")
            send_severity_to_arduino(severity_score)
        
    except sd.PortAudioError as e:
        st.error(f"Error with audio recording: {e}")

# Classification for uploaded files
def classify_audio(file_path):
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    prediction = clf_model.predict(features_scaled)
    result = "Healthy" if prediction[0] == 0 else "Parkinson Diagnosed"
    st.write(f"The audio has been classified as: **{result}**")
    
    if prediction[0] == 1:
        severity_score = estimate_severity(features)
        st.write(f"Estimated Severity Score: **{severity_score:.2f}**")
        send_severity_to_arduino(severity_score)

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio("temp_audio.wav", format="audio/wav")
    classify_audio("temp_audio.wav")

# Close the serial connection when done
if st.button("Close Connection"):
    if 'arduino' in globals():
        arduino.close()
        st.write("Connection to Arduino closed.")
    else:
        st.write("Arduino connection was not established.")
