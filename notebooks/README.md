# Thiết lập và sử dụng notebook

[Về trang dự án](../README.md)

Ba notebook giữ nguyên nội dung gốc. Hướng dẫn này mô tả các thông tin cần chuẩn bị và những cấu hình phải đối chiếu khi chạy trong một tài khoản hoặc phiên Colab mới.

## Tài khoản và API keys

1. Đăng nhập [Modal](https://modal.com/), chọn workspace, mở **Settings → API Tokens** và tạo token. Lưu cả **Token ID** và **Token Secret**.
2. Lấy **Roboflow API key** tại [Workspace Settings → API](https://app.roboflow.com/settings/api). Key phải có quyền với dataset và model cần sử dụng.
3. Tạo token quyền đọc tại [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens) để dùng trong notebook phân tích.

Trong Google Colab, mở **Secrets** ở thanh bên trái, tạo đúng các tên sau và bật **Notebook access**:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `ROBOFLOW_API_KEY`
- `HF_TOKEN`: dùng trong notebook 03.

Notebook 01 nhập hai Modal token qua hộp nhập ẩn `getpass`; notebook 02 và 03 lấy từ Colab Secrets. Không dán giá trị token vào source hoặc giữ chúng trong output trước khi push GitHub.

Nếu cần dùng CLI trên Windows:

```powershell
python -m pip install modal
modal token new
modal token info
```

`token new` mở trình duyệt để đăng nhập; `token info` kiểm tra tài khoản đang sử dụng. Windows và Colab là hai môi trường khác nhau, vì vậy đăng nhập CLI ở máy local không tự xác thực cho Colab. [Modal token CLI](https://modal.com/docs/cli/latest/token).

## Mở notebook trên Google Colab

Vào [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook**, mở file tương ứng và chạy lần lượt các cell. Những cell cài package chuẩn bị thư viện ở runtime Colab; `modal.Image` định nghĩa môi trường container chạy từ xa.

GPU dùng cho tác vụ được khai báo trong `@app.function(...)` bên Modal. Lệnh `nvidia-smi` đầu notebook chỉ kiểm tra GPU của Colab, không xác nhận GPU Modal. Log ở trong hàm từ xa mới phản ánh tài nguyên thực tế dùng để train hoặc suy luận.

## Notebook 01: detector

Sử dụng **L40S**, YOLOv8x, dataset player v12, **50 epochs**, **batch 6**, **imgsz 1280**.

Với Volume mới, sau khi định nghĩa `prepare_dataset` và trước cell train, chạy lời gọi chuẩn bị dữ liệu trong phiên Colab:

```python
with modal.enable_output():
    with app.run():
        prepare_dataset.remote()
```

Dataset và checkpoint lưu trong Volume **`football-yolo-v12`**. Kết quả của mỗi run nằm tại `/runs/<run_name>/`, gồm `weights/best.pt`, `weights/last.pt`, `results.csv`, biểu đồ và ảnh validation.

Notebook gốc có tên run cố định ở phần tải checkpoint và validation. Khi train một run mới, dùng tên run trả về trong log cho các bước này. Muốn xem kết quả cũ thì giữ tên run đã lưu, với điều kiện tài khoản có quyền truy cập Volume tương ứng.

## Notebook 02: keypoint sân

Sử dụng **A100-80GB**, YOLOv8x-pose, dataset pitch v15, **100 epochs**, **batch 16**, **imgsz 640**, **mosaic 0**.

Chạy phần chuẩn bị dataset trước training. Kết quả lưu trong Volume **`pitch-keypoints-v15`**; biến `PITCH_RUN_NAME` xác định run để tải checkpoint và validate. Khi đọc kết quả, phân biệt metric bounding box và metric pose: pose mới đánh giá vị trí keypoint.

Các cell **Deploy model on Roboflow** cuối hai notebook huấn luyện upload checkpoint vào project đích. Đối chiếu workspace, project và version với tài khoản có quyền ghi trước khi chạy. Khi deploy hoàn tất, lấy đúng **Model ID** ở phần Deploy để sử dụng cho inference.

## Notebook 03: phân tích video

Sử dụng **L40S** cho detection và SigLIP. Chạy theo các nhóm cell: cấu hình secrets và môi trường → tải video → detection/tracking → lấy crop và fit phân đội → keypoint/homography → radar, Voronoi và đường đi bóng.

Những cấu hình cần đối chiếu:

- **Video:** `/content/121364_0.mp4` ở Colab; `/inputs/121364_0.mp4` trong Volume `football-ai-videos`.
- **Player model:** `football-players-detection-3zvbc/11`, model công khai đang dùng trong notebook; khác checkpoint fine-tune từ notebook 01.
- **Pitch model:** `nguyns-workspace-do3bt/football-field-detection-f07vi-gh8mi-1-yolov8x-pose-t1`, cần API key có quyền truy cập.
- **SigLIP:** `google/siglip-base-patch16-224`; embedding được dùng cùng UMAP và KMeans để phân đội.

Có thể chạy notebook 03 mà không train lại 01–02 nếu đã có quyền model. Khi thay model sân, thứ tự 32 keypoint phải khớp cấu hình `SoccerPitchConfiguration`. Khi đổi video, thay nhất quán các đường dẫn cố định trong phiên Colab và lấy lại crop để fit phân đội theo trang phục của trận mới.

Notebook 03 gốc kết thúc ở phần đường đi bóng. MP4 trong README là clip được cung cấp riêng; repository không thêm cell render vào notebook gốc.

## Theo dõi phiên chạy

Các notebook gọi hàm bằng `app.run()` và `.remote()`, không cần `modal deploy`. App kết thúc sau khi lời gọi hoàn tất; dữ liệu đã commit vào Volume vẫn tồn tại. Kiểm tra job và mức sử dụng tài nguyên trên dashboard Modal hoặc bằng CLI:

```powershell
modal app list
modal volume list
modal volume ls football-yolo-v12 /runs
modal volume ls pitch-keypoints-v15 /runs
```

Nếu cần dừng một job đang chạy, dùng `modal app stop <app-id>` với ID lấy từ log hoặc `app list`.

## Xử lý lỗi thường gặp

- **Lỗi xác thực:** kiểm tra tên Secrets, Notebook access, token còn hiệu lực và đúng workspace. Quyền đọc dataset không tự bảo đảm quyền truy cập model riêng.
- **Thiếu dataset hoặc `.ready`:** gọi hàm chuẩn bị dữ liệu trong đúng App/Volume trước training.
- **Không có `CUDAExecutionProvider`:** đối chiếu image CUDA và ONNX Runtime trong notebook; chạy bước kiểm tra model trước khi xử lý video.
- **Hết VRAM:** kiểm tra GPU thực tế phía Modal; giảm batch trong phiên thử nghiệm nếu cần và ghi lại cấu hình mới.
- **Không tải được kết quả:** kiểm tra tên Volume, tên run và đường dẫn artifact. Cảnh báo timeout khi chờ log cuối cần được đối chiếu App status và file đã lưu.
- **Homography không đủ điểm:** kiểm tra kết quả keypoint và thứ tự điểm; không suy ra phép chiếu từ các cặp điểm không đủ điều kiện hình học.

Notebook có output ảnh và log thực nghiệm được giữ nguyên. Khi chạy lại, kiểm tra các output chứa thông tin xác thực trước khi commit.
