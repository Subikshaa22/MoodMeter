import sys
import pickle
import numpy as np
import soundfile as sf
import librosa

# Path to the model
model_path = 'model/Emotion_Voice_Detection_Model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Feature extraction function
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        # Ensure the signal length is adequate for FFT
        if len(X) < 2048:
            X = np.pad(X, (0, 2048 - len(X)), mode='constant')  # Pad with zeros
            n_fft = len(X)
        else:
            n_fft = 2048
        
        # Initialize feature array
        result = np.array([])

        # Extract MFCCs
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=n_fft).T, axis=0)
            result = np.hstack((result, mfccs))

        # Extract Chroma features
        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=n_fft))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        # Extract Mel-spectrogram features
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft).T, axis=0)
            result = np.hstack((result, mel))
    
    return result

# Main prediction function
def predict(file_path, mfcc=True, chroma=True, mel=True):
    try:
        features = extract_feature(file_path, mfcc, chroma, mel)
        if features.size == 0:
            raise ValueError("No features were extracted from the audio file.")
        
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return emotions[prediction[0]]
    except Exception as e:
        return f"Error during prediction: File is too big "

# Get the file path from the command line argument
file_path = sys.argv[1]
predicted_emotion = predict(file_path)
print(predicted_emotion)
