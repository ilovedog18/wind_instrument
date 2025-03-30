import streamlit as st
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import os
import datetime
import hashlib

# ---------------- Kết nối vào MySQL -----------------
USERNAME = 'root'
PASSWORD = 'hoang2001'
HOST = 'localhost'
DATABASE = 'music_db'

engine = create_engine(f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}')

# ---------------- Hàm trích xuất đặc trưng ----------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features = np.concatenate((
        np.mean(mfcc, axis=1),
        [np.mean(spec_centroid)],
        [np.mean(spec_rolloff)],
        [np.mean(zcr)],
        np.mean(chroma, axis=1)
    ))
    return features

# ---------------- Hàm tính hash file âm thanh ----------------
def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# ----------------- Tiêu đề ứng dụng ------------------
st.title("🎷 Hệ thống tìm kiếm âm thanh nhạc cụ bộ hơi")

uploaded_file = st.file_uploader("Chọn file âm thanh cần tìm kiếm:", type=['wav'])

if uploaded_file is not None:
    # Tạo tên file duy nhất dựa theo thời gian upload
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{uploaded_file.name}"

    # Đường dẫn lưu file upload
    save_folder = "uploaded_sounds"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    saved_file_path = os.path.join(save_folder, unique_filename)

    # Lưu file vào folder uploaded_sounds
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(saved_file_path, format='audio/wav')

    # Trích xuất đặc trưng và tính hash từ file upload
    input_features = extract_features(saved_file_path)
    uploaded_file_hash = calculate_file_hash(saved_file_path)

    # Truy vấn dữ liệu đặc trưng từ MySQL
    df = pd.read_sql('SELECT * FROM instrument_features', engine)

    # Tính similarity (bỏ qua cột label, file_path, file_hash)
    similarities = cosine_similarity([input_features], df.iloc[:, 1:-3])
    df['similarity'] = similarities[0]

    # Loại bỏ file vừa upload khỏi kết quả tìm kiếm
    df_result = df[df['file_hash'] != uploaded_file_hash]

    # Bỏ các file trùng nhau, chỉ giữ lại 1 file mỗi hash
    df_result_unique = df_result.drop_duplicates(subset=['file_hash'], keep='first', ignore_index=True)

    # Nếu kết quả ít hơn 3, bổ sung từ các file đã lọc để luôn đủ 3 kết quả
    if len(df_result_unique) < 3:
        needed = 3 - len(df_result_unique)
        df_remaining = df_result[~df_result.index.isin(df_result_unique.index)]
        df_result_unique = pd.concat([df_result_unique, df_remaining.head(needed)], ignore_index=True)

    # Lấy top 3 kết quả tương đồng
    top3 = df_result_unique.sort_values(by='similarity', ascending=False).head(3)

    # ----------------- Hiển thị kết quả tìm kiếm ------------------
    st.subheader("🎵 Top 3 file âm thanh tương đồng nhất:")
    #in ra  giao diện ( tên file , nhãn tên loại nhạc cụ ) nếu ko cần thì comment
    for idx, row in top3.iterrows():
        # st.markdown(f"### 🎶 **{row['filename']}**")
        # st.write(f"**Loại nhạc cụ:** {row['label']}")
        st.write(f"**Độ tương đồng:** {row['similarity']:.4f}")

        audio_played = False  # Biến kiểm tra đã phát âm thanh chưa

        # Kiểm tra đường dẫn DB tồn tại và hợp lệ trước
        audio_path_db = row['file_path']
        if pd.notnull(audio_path_db) and os.path.exists(audio_path_db):
            audio_file = open(audio_path_db, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            audio_played = True

        # Nếu chưa phát được thì kiểm tra đường dẫn mặc định
        if not audio_played:
            audio_path_default = f"wind_instruments/{row['filename']}"
            if os.path.exists(audio_path_default):
                audio_file = open(audio_path_default, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')

    # ----------------- Lưu file mới vào DB (không hiện thông báo) ------------------
    new_entry = pd.DataFrame({
        'filename': [uploaded_file.name],
        'mfcc_1': [input_features[0]], 'mfcc_2': [input_features[1]], 'mfcc_3': [input_features[2]],
        'mfcc_4': [input_features[3]], 'mfcc_5': [input_features[4]], 'mfcc_6': [input_features[5]],
        'mfcc_7': [input_features[6]], 'mfcc_8': [input_features[7]], 'mfcc_9': [input_features[8]],
        'mfcc_10': [input_features[9]],'mfcc_11': [input_features[10]],'mfcc_12': [input_features[11]],
        'mfcc_13': [input_features[12]],
        'spec_centroid': [input_features[13]],
        'spec_rolloff': [input_features[14]],
        'zcr': [input_features[15]],
        'chroma_1': [input_features[16]], 'chroma_2': [input_features[17]], 'chroma_3': [input_features[18]],
        'chroma_4': [input_features[19]], 'chroma_5': [input_features[20]], 'chroma_6': [input_features[21]],
        'chroma_7': [input_features[22]], 'chroma_8': [input_features[23]], 'chroma_9': [input_features[24]],
        'chroma_10': [input_features[25]],'chroma_11': [input_features[26]],'chroma_12': [input_features[27]],
        'label': [top3.iloc[0]['label']],
        'file_path': [saved_file_path],
        'file_hash': [uploaded_file_hash]
    })

    new_entry.to_sql('instrument_features', engine, if_exists='append', index=False)
