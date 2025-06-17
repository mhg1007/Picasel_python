from ultralytics import YOLO

model=YOLO('yolo11s.pt')

train_results=model.train(
    data="data.yaml",
    epochs=15,
    imgsz=1280,
    batch=1,
    cache="disk",
    device="cpu",
)

metrics=model.val()

path=model.export(format="onnx", nms=True, dynamic=True, simplify=True)