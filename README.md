# Content-Based Image Retrieval (CBIR) System

Hệ thống Tìm kiếm Ảnh dựa trên Nội dung (CBIR) được phát triển nhằm tự động hóa quá trình truy xuất hình ảnh dựa trên các đặc trưng trực quan cấp độ thấp, kết hợp pipeline đa đặc trưng (HSV, LBP, HOG), chuẩn hóa vector L2 và thuật toán khoảng cách Euclidean.

 👥 Thành viên nhóm thực hiện
* **Bùi Đức Việt** (Trưởng nhóm) – MSSV: K24DTCN295
* **Phan Phúc Thịnh** – MSSV: K24DTCN295
* **Lương Thành Đạt** – MSSV: K24DTCN260

---

## 📂 Cấu trúc thư mục dự án
```text
CBIR_Project/
│
├── .venv/              # Môi trường ảo Python (Virtual Environment)
├── dataset/            # Thư mục chứa cơ sở dữ liệu ảnh gốc dùng để tra cứu
├── query_images/       # Thư mục chứa ảnh truy vấn (test images)
├── app.py              # Giao diện ứng dụng trực quan (Xây dựng bằng Streamlit)
├── main.py             # Mã nguồn cốt lõi (Pipeline trích xuất đặc trưng và tính toán khoảng cách)
└── requirements.txt    # Danh sách các thư viện Python phụ thuộc

```

### ⚙️ Hướng dẫn cài đặt và chạy chương trình

1. Cài đặt các thư viện phụ thuộc
Mở Terminal tại thư mục gốc của dự án (CBIR_Project) và chạy lệnh sau để cài đặt môi trường:

Bash
pip install -r requirements.txt
2. Chuẩn bị dữ liệu
Đưa các hình ảnh cơ sở dữ liệu vào thư mục dataset/.

Đặt các ảnh cần kiểm thử vào thư mục query_images/.

3. Chạy ứng dụng giao diện web (Streamlit)
Khởi chạy giao diện trực quan để tương tác, tải ảnh truy vấn và xem kết quả Top-5 tương đồng:

Bash
streamlit run app.py
Sau đó truy cập đường dẫn hiển thị trên terminal (ví dụ: http://localhost:8501) trên trình duyệt web.

4. Chạy kiểm thử thuật toán cốt lõi
Để chạy chương trình xử lý dòng lệnh hoặc kiểm tra đánh giá định lượng Precision@K:

Bash
python main.py