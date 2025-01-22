import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from skimage import io, transform, color
from skimage.metrics import peak_signal_noise_ratio as psnr

scale = 2  
patch_size = 41  
batch_size = 64
epochs = 100

# Function to create low-resolution images
def create_lr_hr_pair(image, scale):
    hr = transform.resize(image, (image.shape[0] // scale, image.shape[1] // scale), anti_aliasing=True)
    lr = transform.resize(hr, (image.shape[0], image.shape[1]), anti_aliasing=True)
    return lr, image

# Function to extract patches from images
def extract_patches(lr, hr, patch_size, scale):
    lr_patches = []
    hr_patches = []
    for i in range(0, lr.shape[0] - patch_size + 1, patch_size):
        for j in range(0, lr.shape[1] - patch_size + 1, patch_size):
            lr_patch = lr[i:i + patch_size, j:j + patch_size]
            hr_patch = hr[i:i + patch_size, j:j + patch_size]
            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)
    return np.array(lr_patches), np.array(hr_patches)

# Function to load images and create dataset
def load_dataset(base_path, patch_size, scale):
    lr_patches = []
    hr_patches = []
    for img_name in os.listdir(base_path):
        img_path = os.path.join(base_path, img_name)
        image = io.imread(img_path)
        if image.ndim == 3:
            image = color.rgb2ycbcr(image)[:, :, 0]  # Use Y channel for processing
        lr, hr = create_lr_hr_pair(image, scale)
        lr_patch, hr_patch = extract_patches(lr, hr, patch_size, scale)
        lr_patches.extend(lr_patch)
        hr_patches.extend(hr_patch)
    return np.array(lr_patches), np.array(hr_patches)

# Load the dataset
train_path = 'BSDS300/images/train'
test_path = 'BSDS300/images/test'
lr_train, hr_train = load_dataset(train_path, patch_size, scale)
lr_test, hr_test = load_dataset(test_path, patch_size, scale)

# Normalize the data
lr_train = lr_train / 255.0
hr_train = hr_train / 255.0
lr_test = lr_test / 255.0
hr_test = hr_test / 255.0

# Define the VDSR model
def build_vdsr():
    inputs = layers.Input(shape=(patch_size, patch_size, 1))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    for _ in range(18):
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, (3, 3), padding='same')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Build and train the model
vdsr_model = build_vdsr()
vdsr_model.fit(lr_train, hr_train, batch_size=batch_size, epochs=epochs, validation_data=(lr_test, hr_test))

# Save the model
vdsr_model.save('V.h5')

# Function to calculate average PSNR score
def calculate_average_psnr(model, lr_images, hr_images):
    total_psnr = 0.0
    num_images = len(lr_images)
    for i in range(num_images):
        lr = lr_images[i].reshape(1, patch_size, patch_size, 1)
        hr = hr_images[i].reshape(patch_size, patch_size)
        sr = model.predict(lr, verbose=0).reshape(patch_size, patch_size)
        total_psnr += psnr(hr, sr, data_range=1.0)
    return total_psnr / num_images

# Calculate and print average PSNR score
average_psnr = calculate_average_psnr(vdsr_model, lr_test, hr_test)
print(f"Average PSNR: {average_psnr:.2f} dB")