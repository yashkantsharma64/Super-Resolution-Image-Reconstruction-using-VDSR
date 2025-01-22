from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from skimage import color, io
from skimage.metrics import peak_signal_noise_ratio as psnr
import math

# Function to load and preprocess a user-provided image
def load_user_image(image_path):
    image = io.imread(image_path)
    return image

def sliding_window_super_resolve(model, lr_image, patch_size, stride, batch_size):
    h, w = lr_image.shape[:2]
    c = 1 if lr_image.ndim == 2 else lr_image.shape[2]
    
    # Calculate necessary padding
    pad_h = (math.ceil((h - patch_size) / stride) + 1) * stride + patch_size - h
    pad_w = (math.ceil((w - patch_size) / stride) + 1) * stride + patch_size - w

    # Pad the image to ensure full coverage
    padded_image = np.pad(
        lr_image,
        ((0, pad_h), (0, pad_w), (0, 0)) if c > 1 else ((0, pad_h), (0, pad_w)),
        mode="reflect"
    )
    padded_h, padded_w = padded_image.shape[:2]

    output = np.zeros((padded_h, padded_w, c) if c > 1 else (padded_h, padded_w))
    count_map = np.zeros_like(output)

    patches = []
    patch_coords = []

    # Extract patches for processing
    for i in range(0, padded_h - patch_size + 1, stride):
        for j in range(0, padded_w - patch_size + 1, stride):
            lr_patch = padded_image[i:i + patch_size, j:j + patch_size]
            patches.append(lr_patch)
            patch_coords.append((i, j))

            if len(patches) == batch_size:
                # Process patches in batches
                batch = np.array(patches).astype(np.float32) / 255.0
                sr_patches = model.predict(batch)
                for k, (x, y) in enumerate(patch_coords):
                    sr_patch = sr_patches[k].squeeze() * 255.0
                    if c == 1:
                        output[x:x + patch_size, y:y + patch_size] += sr_patch
                    else:
                        output[x:x + patch_size, y:y + patch_size, :] += sr_patch
                    count_map[x:x + patch_size, y:y + patch_size] += 1
                patches = []
                patch_coords = []

    # Process remaining patches
    if patches:
        batch = np.array(patches).astype(np.float32) / 255.0
        sr_patches = model.predict(batch)
        for k, (x, y) in enumerate(patch_coords):
            sr_patch = sr_patches[k].squeeze() * 255.0
            if c == 1:
                output[x:x + patch_size, y:y + patch_size] += sr_patch
            else:
                output[x:x + patch_size, y:y + patch_size, :] += sr_patch
            count_map[x:x + patch_size, y:y + patch_size] += 1

    # Normalize output by count_map to avoid overlapping intensity buildup
    output = np.divide(output, count_map, out=np.zeros_like(output), where=count_map != 0)

    # Crop the image back to its original size
    output = output[:h, :w] if c == 1 else output[:h, :w, :]
    return np.clip(output, 0, 255).astype(np.uint8)

# Function to super resolve a user-provided image
def super_resolve_user_image(model_path, image_path, patch_size, stride, batch_size=64):
    model = tf.keras.models.load_model(model_path)
    lr_image = load_user_image(image_path)
    lr_ycbcr = color.rgb2ycbcr(lr_image)
    lr_y = lr_ycbcr[:, :, 0]  
    sr_y = sliding_window_super_resolve(model, lr_y, patch_size, stride, batch_size)
    sr_ycbcr = lr_ycbcr.copy()
    sr_ycbcr[:, :, 0] = sr_y 
    sr_image = color.ycbcr2rgb(sr_ycbcr)
    psnr_value = psnr(lr_y, sr_y, data_range=255)
    return sr_image, psnr_value

app = Flask(__name__)
CORS(app)  
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    model_path = 'V.h5' 
    patch_size = 41
    stride = 21
    try:
        sr_image, psnr_value = super_resolve_user_image(model_path, filepath, patch_size, stride)
        result_filename = f"sr_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        io.imsave(result_path, np.clip(sr_image * 255.0, 0, 255).astype(np.uint8))  
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'original': filepath,
        'result': result_path,
        'psnr': psnr_value
    })

#Serve uploaded and processed images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
