import streamlit as st
import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, hog
from PIL import Image

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hệ Thống CBIR Nâng Cao (Color + LBP + HOG)", layout="wide", page_icon="🔍")

np.random.seed(42)


# --- 1. TRÍCH XUẤT ĐẶC TRƯNG MÀU SẮC (HSV) ---
def extract_color_histogram(image, bins=(8, 12, 3)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# --- 2. TRÍCH XUẤT ĐẶC TRƯNG KẾT CẤU (LBP) ---
def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


# --- 3. TRÍCH XUẤT ĐẶC TRƯNG HÌNH DẠNG (HOG) ---
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    hog_feat = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualize=False)
    return hog_feat


# --- PIPELINE KẾT HỢP VÀ CHUẨN HÓA L2 ---
def extract_combined_features(image):
    image = cv2.resize(image, (256, 256))

    # 1. Trích xuất và chuẩn hóa Color Histogram
    color_feat = extract_color_histogram(image)
    color_feat = color_feat / (np.linalg.norm(color_feat) + 1e-7)

    # 2. Trích xuất và chuẩn hóa LBP
    lbp_feat = extract_lbp_features(image)
    lbp_feat = lbp_feat / (np.linalg.norm(lbp_feat) + 1e-7)

    # 3. Trích xuất và chuẩn hóa HOG
    hog_feat = extract_hog_features(image)
    hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-7)

    # Ghép nối các đặc trưng đã chuẩn hóa
    return np.concatenate([color_feat, lbp_feat, hog_feat])


@st.cache_data
def build_database_features(dataset_dir):
    features_db = {}
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        return features_db

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(dataset_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                feat = extract_combined_features(img)
                features_db[filename] = feat
    return features_db


# --- GIAO DIỆN WEB STREAMLIT ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🔍 Hệ Thống CBIR Nâng Cao</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Đồ án Xử lý ảnh - Kết hợp: Màu sắc (HSV) + Kết cấu (LBP) + Hình dạng (HOG)</p>",
    unsafe_allow_html=True)
st.write("---")

db_path = "./dataset"
db_features = build_database_features(db_path)

st.sidebar.header("⚙️ Tùy chọn truy vấn")
uploaded_file = st.sidebar.file_uploader("Tải ảnh mẫu từ máy tính...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    query_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 📥 Ảnh Truy Vấn")
        st.image(image, caption="Query Image", use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top-5 Kết Quả Tương Đồng Nhất")
        if len(db_features) == 0:
            st.warning("⚠️ Thư mục dataset đang trống! Hãy thêm ảnh vào thư mục 'dataset'.")
        else:
            query_feat = extract_combined_features(query_img_cv)
            scores = {}
            for filename, db_feat in db_features.items():
                distance = np.linalg.norm(query_feat - db_feat)
                scores[filename] = distance

            sorted_scores = sorted(scores.items(), key=lambda item: item[1])[:5]

            cols = st.columns(5)
            for idx, (fname, score) in enumerate(sorted_scores):
                img_path = os.path.join(db_path, fname)
                result_img = Image.open(img_path)
                with cols[idx]:
                    st.image(result_img, caption=f"Top {idx + 1}\nDist: {score:.2f}", use_container_width=True)
else:
    st.info("👈 **Hướng dẫn:** Hãy tải một bức ảnh bất kỳ lên từ thanh công cụ bên trái để hệ thống tiến hành tìm kiếm!")