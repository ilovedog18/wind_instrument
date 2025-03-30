import pandas as pd
from sqlalchemy import create_engine

# Cấu hình kết nối MySQL của bạn
USERNAME = ''         # Thay bằng username MySQL của bạn
PASSWORD = '' # Thay bằng mật khẩu MySQL của bạn
HOST = ''        # thường mặc định là localhost
DATABASE = ''     # Database bạn đã tạo trong MySQL

# Kết nối vào CSDL MySQL
connection_str = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}'
engine = create_engine(connection_str)

# Đọc dữ liệu đặc trưng từ CSV
df = pd.read_csv('instrument_features.csv')

# Ghi dữ liệu vào bảng MySQL, bảng đặt tên 'instrument_features'
df.to_sql('instrument_features', con=engine, if_exists='replace', index=False)

print("✅ Lưu dữ liệu vào MySQL thành công!")
