from ultralytics import YOLO
import torch

def main(model_name, dataset_name):

    device = torch.cuda.current_device()
    # Load a model
    model = YOLO(model_name)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=dataset_name, epochs=100, imgsz=640, device = device)

if __name__ == "__main__":
    model = "yolov8x.pt"
    dataset = "my_dataset.yaml"

    main(model, dataset)