from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

def pencil_sketch(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = 255 - gray_image

    # Blur the inverted image
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)

    # Invert the blurred image back
    inverted_blurred = 255 - blurred_image

    # Create the pencil sketch image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    return pencil_sketch

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read image file
    npimg = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Ensure the image is in RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate pencil sketch
    sketch = pencil_sketch(img_rgb)

    # Convert sketch to JPEG format
    is_success, buffer = cv2.imencode(".jpg", sketch)

    if not is_success:
        return "Conversion failed", 500

    # Convert to BytesIO object
    io_buf = BytesIO(buffer)

    # Return sketch as downloadable file
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
