import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Extract MFCCs
def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Shape: (Time, n_mfcc)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(40, 100, 1)),  # 100 frames
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')  # 8 emotions
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'