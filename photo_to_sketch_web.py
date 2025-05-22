import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn

# Optional: Shallow CNN (random weights for demo)
class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 1, 3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ShallowCNN().to(device)
cnn_model.eval()

def photo_to_sketch(image, use_cnn=False):
    # Convert PIL image to OpenCV
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Optional: CNN enhancement
    if use_cnn:
        input_img = gray.astype('float32') / 255.0
        input_img = np.expand_dims(input_img, axis=(0, 1))  # (1, 1, H, W)
        input_tensor = torch.from_numpy(input_img).to(device)
        with torch.no_grad():
            output = cnn_model(input_tensor)
        cnn_output = output.squeeze().cpu().numpy() * 255
        cnn_output = cnn_output.astype('uint8')
        gray = cnn_output  # Use CNN output as new gray

    # Invert the blurred image
    inverted_blur = 255 - blurred

    # Dodge blend
    def dodgeV2(image, mask):
        return cv2.divide(image, 255 - mask, scale=256)

    sketch = dodgeV2(gray, inverted_blur)
    return sketch

iface = gr.Interface(
    fn=photo_to_sketch,
    inputs=[
        gr.Image(type="pil", label="Upload Photo"),
        gr.Checkbox(label="Use Shallow CNN Enhancement (Demo Only)")
    ],
    outputs=gr.Image(type="numpy", label="Pencil Sketch"),
    title="Photo to Sketch Converter",
    description="Upload a photo to convert it into a pencil sketch using edge detection and (optionally) a shallow CNN."
)

if __name__ == "__main__":
    iface.launch()