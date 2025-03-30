import os
import pandas as pd
import librosa
import numpy as np
import hashlib

# Hàm trích xuất đặc trưng âm thanh
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    # Spectral Roll-off
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    spec_rolloff_mean = np.mean(spec_rolloff)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Ghép toàn bộ thuộc tính lại với nhau
    features = np.concatenate((
        mfcc_mean,
        [spec_centroid_mean],
        [spec_rolloff_mean],
        [zcr_mean],
        chroma_mean
    ))

    return features

# Hàm tính hash cho file
def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Đường dẫn tới thư mục chứa âm thanh
folder_path = r"C:\Users\hoang\Desktop\music media\wind_instruments"
data = []

# Lặp qua toàn bộ các file
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        file_hash = calculate_file_hash(file_path)

        # Đoạn gán nhãn hoàn chỉnh
        file_lower = file.lower()
        if '[flu]' in file_lower:
            label = 'Flute'
        elif '[sax]' in file_lower:
            label = 'Saxophone'
        elif '[cla]' in file_lower:
            label = 'Clarinet'
        elif 'flute' in file_lower:
            label = 'Flute'
        elif 'saxophone' in file_lower or 'sax' in file_lower:
            label = 'Saxophone'
        elif 'clarinet' in file_lower:
            label = 'Clarinet'
        else:
            label = 'Unknown'

        # Lưu đặc trưng, nhãn, file_path, file_hash
        data.append([file] + features.tolist() + [label, file_path, file_hash])

# Tạo DataFrame đầy đủ
columns = ['filename'] + [f'mfcc_{i}' for i in range(1,14)] + \
          ['spec_centroid', 'spec_rolloff', 'zcr'] + \
          [f'chroma_{i}' for i in range(1,13)] + \
          ['label', 'file_path', 'file_hash']

df = pd.DataFrame(data, columns=columns)

# Lưu vào file CSV
df.to_csv('instrument_features.csv', index=False)

print("Hoàn tất trích xuất đặc trưng và lưu file instrument_features.csv!")
