# ✏️ Photo to Sketch Converter
Convert any photograph into a realistic pencil sketch using OpenCV edge blending techniques. Optionally enhance the sketch effect using a demo Shallow CNN.

🌟 Features

🖼️ Upload any photo (JPEG/PNG)

✏️ Converts photo to pencil sketch using grayscale + dodge blending

🧠 Optional CNN enhancement for sketch realism (demo purpose only)

⚡ Fast, browser-based experience powered by Gradio

📁 Project Files

photo-to-sketch/
│
├── photo_to_sketch_web.py       # Gradio app script
├── img2sketch.ipynb             # Jupyter notebook version (exploration/training)

🧠 About the CNN

The CNN used here is a simple two-layer convolutional network.

It serves as a demo and is initialized with random weights (not trained).

It slightly alters the sketching process if enabled, but is not critical for the sketch output.

🖼️ How It Works

Convert to Grayscale

Apply Gaussian Blur

Invert the Blurred Image

Apply Dodge Blend:

Final sketch is generated with:

sketch = divide(gray, 255 - blurred_inverted)

🧪 Usage

Option 1: Run Locally

1. Clone the repository:

git clone https://github.com/yourusername/photo-to-sketch.git

cd photo-to-sketch

2. Install dependencies:

pip install gradio opencv-python torch numpy

3. Launch the app:

python photo_to_sketch_web.py

The app will open in your browser at:
http://127.0.0.1:7860

🧪 Option 2: Jupyter Notebook

You can also explore and run sketching logic inside the provided notebook:

img2sketch.ipynb

This is useful for:

Visualizing intermediate steps

Experimenting with CNN layers

Fine-tuning edge detection

🧾 Notes

CNN is purely optional and untrained — included as a framework for expansion.

Works best with clear portrait or object images.

Accepts .jpg, .png, and .jpeg file types.

🛠 Future Improvements

Train the CNN on edge-detection datasets for enhancement.

Add batch processing support.

Add stylized sketching (e.g., charcoal, ink, comic).
