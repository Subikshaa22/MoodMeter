# Emotion Detection from Audio Files 

## Overview

This project is designed to detect emotions from audio files in `.wav` format. The project leverages machine learning techniques to analyze speech and predict the emotion conveyed. The system is built using Python for the backend processing and TensorFlow/Keras for the machine learning model. The frontend is a simple web interface built with Express.js and EJS to upload audio files and display the predicted emotion.

## Dataset 

The dataset used for this project is the RAVDESS dataset.It has a total of 1440 audio files recorded by 24 actors(12 male and 12 female).Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav).
Filename identifiers:

- **Modality (01 = full-AV, 02 = video-only, 03 = audio-only).**
- **Vocal channel (01 = speech, 02 = song).**
- **Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).**
- **Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.**
- **Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").**
- **Repetition (01 = 1st repetition, 02 = 2nd repetition).**
- **Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).**


## Project Structure

- **MoodMeter.ipynb**: A Jupyter Notebook that preprocesses the audio data, trains a Multi-Layer Perceptron (MLP) classifier, and saves the trained model.
- **predict.py**: A Python script that loads the trained model and predicts the emotion from a given audio file.
- **server.js**: An Express.js server that handles file uploads, runs the prediction script, and returns the result to the client.
- **index.ejs**: An EJS template that serves as the frontend, allowing users to upload audio files and see the predicted emotion.

## Features

- **Emotion Detection**: Detects emotions such as calm, happy, fearful, and disgust from audio files.
- **Preprocessing**: Cleans and preprocesses audio data using `librosa` and other libraries.
- **Model Training**: Trains an MLP classifier using extracted audio features like MFCC, chroma, and mel-spectrogram.
- **Web Interface**: Allows users to upload `.wav` files and view the predicted emotion through a simple web interface.

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `librosa`
- `matplotlib`
- `tensorflow`
- `keras`
- `scipy`
- `soundfile`
- `tqdm`
- `scikit-learn`
- `python-speech-features`
- `pydub`
- `noisereduce`
- `pickle`
- `express`
- `express-fileupload`
- `ejs`
- `child_process`

## Usage

### 1. Training the Model

Run the `MoodMeter.ipynb` notebook to preprocess the audio data, train the MLP classifier, and save the model as `Emotion_Voice_Detection_Model.pkl`.

### 2. Running the Server

Start the server using:

```bash
node server.js
```

### 3. Making Predictions

You can also directly predict the emotion of an audio file by running the `predict.py` script:
(The audio file is required to be atmost 1 second long.(85KB))

```bash
python predict.py path_to_audio_file.wav
```

### 4. Frontend Interface

The index.ejs file provides a simple interface for uploading audio files and displaying the prediction result.

## Observed Emotions

The project currently observes the following emotions:

- Calm
- Happy
- Fearful
- Disgust

You can modify the `observed_emotions` dictionary in the code to change or add more emotions.

### Challenges Faced

- **Noise and Distortions**: Handling background noise and distortions in audio files required advanced preprocessing techniques like noise reduction and filtering.

- **Dimensionality and Complexity**: Extracting meaningful features from audio, such as MFCC, chroma, and mel-spectrograms, involved managing high-dimensional data and ensuring computational efficiency.

- **Overfitting**: Ensuring that the model generalizes well to unseen data without overfitting on the training set required careful tuning of hyperparameters and regularization techniques.

- **File Handling**: Managing file uploads and ensuring correct file paths for processing in the web application required careful handling.

- **Emotion Classification**: Achieving high accuracy in classifying subtle emotional differences in speech can be challenging, given the variability in human speech patterns and the complexity of emotions.

### Future Scope and Improvements

- **Deep Learning Models**: Explore advanced deep learning architectures like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for more effective feature learning and classification.

- **Additional Emotions**: Expand the dataset to include a wider range of emotions or more nuanced emotional states for a more comprehensive model.
   
- **Cross-Language and Cross-Cultural Data**: Incorporate data from different languages and cultures to make the model more versatile and applicable globally.

- **Latency Reduction**: Optimize the system for real-time processing to enable immediate emotion detection in live applications.
  
- **Video, Text, and Image Analysis**: Extend the system to include analysis of video, text, and images alongside audio for a more holistic approach to emotion detection and understanding.



