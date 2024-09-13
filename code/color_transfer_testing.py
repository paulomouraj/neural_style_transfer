
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_statistics(image):
    """ Compute the mean and covariance of the LAB channels. """
    reshaped_image = image.reshape(-1, 3)
    mean = np.mean(reshaped_image, axis=0)
    covariance = np.cov(reshaped_image, rowvar=False)
    return mean, covariance

def color_transfer(style_image, content_image):
    # Convert images from RGB to LAB
    style_lab = cv2.cvtColor(style_image, cv2.COLOR_RGB2LAB)
    content_lab = cv2.cvtColor(content_image, cv2.COLOR_RGB2LAB)
    
    # Calculate means and covariances for style and content images in LAB
    mu_s, sigma_s = compute_statistics(style_lab)
    mu_c, sigma_c = compute_statistics(content_lab)

    # Perform Cholesky decomposition on content covariance matrix
    chol_c = np.linalg.cholesky(sigma_c)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_s)
    sigma_s_half = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    A = chol_c @ np.linalg.inv(sigma_s_half)

    # Calculate the translation vector b
    b = mu_c - A @ mu_s

    # Apply the transformation to each pixel
    transformed_lab = (style_lab.reshape(-1, 3) @ A.T + b).clip(0, 255).astype(np.uint8)
    transformed_lab = transformed_lab.reshape(style_lab.shape)

    # Convert transformed LAB back to RGB
    transformed_image = cv2.cvtColor(transformed_lab, cv2.COLOR_LAB2RGB)

    return transformed_image

def plot_histogram(image, title, ax):
    """ Plot the histogram for each RGB channel. """
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_xlim([0, 256])
    ax.set_title(title)

def load_and_transform_images(style_path, content_path):
    # Load images
    style_image = cv2.imread(style_path)
    content_image = cv2.imread(content_path)
    
    # Dummy transformation: convert style image color space as an example
    transformed_image = color_transfer(style_image, content_image)
    
    # Convert images to RGB for display and histogram calculation
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    
    return style_image, content_image, transformed_image

# Paths to the images
# Load and transform images
# style_image, content_image, transformed_image = load_and_transform_images(style_url, content_url)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot images
axs[0, 0].imshow(style_image)
axs[0, 0].set_title('Style Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(content_image)
axs[0, 1].set_title('Content Image')
axs[0, 1].axis('off')

axs[0, 2].imshow(transformed_image)
axs[0, 2].set_title('Transformed Style Image')
axs[0, 2].axis('off')

# Plot histograms
plot_histogram(style_image, 'Style Histogram', axs[1, 0])
plot_histogram(content_image, 'Content Histogram', axs[1, 1])
plot_histogram(transformed_image, 'Transformed Histogram', axs[1, 2])

# Adjust layout
plt.tight_layout()
plt.show()
