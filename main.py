import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt

# 1. Cố định hạt giống ngẫu nhiên theo yêu cầu kỹ thuật
np.random.seed(42)


def extract_color_histogram(image, bins=(8, 12, 3)):
    """Chương 3: Trích xuất đặc trưng màu sắc trong không gian HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_lbp_features(image):
    """Chương 6: Trích xuất đặc trưng kết cấu (Local Binary Patterns - LBP)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_hog_features(image):
    """Trích xuất đặc trưng hình dạng (Histogram of Oriented Gradients - HOG)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    hog_feat = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualize=False)
    return hog_feat


def extract_combined_features(image_path):
    """Pipeline kết hợp toàn diện (Màu + LBP + HOG) và chuẩn hóa L2"""
    image = cv2.imread(image_path)
    if image is None:
        return None

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
    combined_feat = np.concatenate([color_feat, lbp_feat, hog_feat])
    return combined_feat


def build_database_features(dataset_dir):
    """Xây dựng kho chứa đặc trưng cho toàn bộ thư mục database"""
    features_db = {}
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        return features_db

    filenames = os.listdir(dataset_dir)
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(dataset_dir, filename)
            feat = extract_combined_features(path)
            if feat is not None:
                features_db[filename] = feat

    print(f"Đã trích xuất thành công đặc trưng kết hợp (Màu + LBP + HOG) cho {len(features_db)} ảnh trong database!")
    return features_db


def search_images(query_path, features_db, top_k=5):
    """Tính khoảng cách Euclidean (L2 Norm) và trả về Top-K"""
    query_feat = extract_combined_features(query_path)
    if query_feat is None:
        return []

    scores = {}
    for filename, db_feat in features_db.items():
        distance = np.linalg.norm(query_feat - db_feat)
        scores[filename] = distance

    sorted_scores = sorted(scores.items(), key=lambda item: item[1])
    return sorted_scores[:top_k]


def evaluate_precision_at_k(features_db, test_queries, k=5):
    """
    Đánh giá định lượng chỉ số Precision@K
    """
    total_precision = 0.0
    num_queries = len(test_queries)

    if num_queries == 0:
        return 0.0

    for query_path, relevant_prefix in test_queries:
        results = search_images(query_path, features_db, top_k=k)
        relevant_count = 0
        for fname, _ in results:
            # Kiểm tra xem tên file kết quả có bắt đầu bằng tiền tố đúng chủ đề không
            if fname.startswith(relevant_prefix):
                relevant_count += 1

        precision = relevant_count / k
        total_precision += precision

    mean_precision = total_precision / num_queries
    print(f"\n📊 Đánh giá định lượng hệ thống - Precision@{k}: {mean_precision:.4f} ({mean_precision * 100:.2f}%)")
    return mean_precision


# --- CHẠY THỬ NGHIỆM VÀ ĐÁNH GIÁ ---
if __name__ == "__main__":
    db_path = "./dataset"
    query_img_path = "./query_images/test.jpg"

    # 1. Xây dựng database đặc trưng
    db_features = build_database_features(db_path)

    # 2. Truy xuất thử nghiệm 1 ảnh và vẽ biểu đồ trực quan
    if os.path.exists(query_img_path):
        results = search_images(query_img_path, db_features, top_k=5)
        print(f"\nKết quả Top-5 ảnh tìm được cho {query_img_path}:")

        plt.figure(figsize=(14, 6))

        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 6, 1)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis('off')

        for rank, (fname, score) in enumerate(results, 1):
            print(f"{rank}. {fname} - Khoảng cách L2: {score:.4f}")

            img_path = os.path.join(db_path, fname)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 6, rank + 1)
                plt.imshow(img)
                plt.title(f"Top {rank}\nDist: {score:.2f}", fontsize=8)
                plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print(f"\nHãy bỏ ảnh vào thư mục 'dataset' và ảnh 'test.jpg' vào 'query_images' để test.")

    # 3. Chạy đánh giá Precision@5 tự động (nếu bạn có danh sách test queries)
    # Ví dụ mẫu: định nghĩa các cặp (đường dẫn ảnh test, tiền tố tên file đúng trong dataset)
    test_set = [
        # ("./query_images/cat_test1.jpg", "cat"),
        # ("./query_images/flower_test1.jpg", "flower")
    ]
    if len(test_set) > 0:
        evaluate_precision_at_k(db_features, test_set, k=5)