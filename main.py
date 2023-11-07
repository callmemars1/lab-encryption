import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = 'img.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))


def generate_s_box():
    s_box = list(range(256))
    np.random.shuffle(s_box)
    return s_box


s_boxes = [
    # S-box for round 1
    [generate_s_box() for _ in range(4)],
    # S-box for round 2
    [generate_s_box() for _ in range(4)],
]
p_box = [1, 0, 3, 2]  # Example P-box


def sp_encrypt(plaintext, s_boxes, p_box):
    # Convert the plaintext image to a 1D array
    plaintext_array = plaintext.flatten()

    # Determine the number of rounds based on the number of S-box sets
    num_rounds = len(s_boxes)

    # Initialize the ciphertext array
    ciphertext_array = np.zeros_like(plaintext_array)

    # Iterate through the plaintext array in chunks of block_size
    block_size = len(p_box)
    for i in range(0, len(plaintext_array), block_size):
        block = plaintext_array[i:i + block_size]

        # Perform the SP-network encryption for each round
        for round_idx in range(num_rounds):
            # Apply the S-boxes
            for j in range(block_size):
                s_box = s_boxes[round_idx][j % len(s_boxes[round_idx])]
                block[j] = s_box[block[j]]

            # Apply the P-box
            block = block[p_box]

        # Store the encrypted block in the ciphertext array
        ciphertext_array[i:i + block_size] = block

    # Convert the ciphertext array back to a 2D image
    ciphertext_image = np.reshape(ciphertext_array, plaintext.shape)

    return ciphertext_image


def sp_decrypt(ciphertext, s_boxes, p_box):
    # Convert the ciphertext image to a 1D array
    ciphertext_array = ciphertext.flatten()

    # Determine the number of rounds based on the number of S-box sets
    num_rounds = len(s_boxes)

    # Initialize the plaintext array
    plaintext_array = np.zeros_like(ciphertext_array)

    # Create the inverse P-box
    inv_p_box = [0] * len(p_box)
    for i, val in enumerate(p_box):
        inv_p_box[val] = i

    # Iterate through the ciphertext array in chunks of block_size
    block_size = len(p_box)
    for i in range(0, len(ciphertext_array), block_size):
        block = ciphertext_array[i:i + block_size]

        # Perform the SP-network decryption for each round
        for round_idx in reversed(range(num_rounds)):
            # Apply the inverse P-box
            block = block[inv_p_box]

            # Apply the inverse S-boxes
            for j in range(block_size):
                inv_s_box = {v: k for k, v in enumerate(s_boxes[round_idx][j % len(s_boxes[round_idx])])}
                block[j] = inv_s_box[block[j]]

        # Store the decrypted block in the plaintext array
        plaintext_array[i:i + block_size] = block

    # Convert the plaintext array back to a 2D image
    plaintext_image = np.reshape(plaintext_array, ciphertext.shape)

    return plaintext_image


def correlation_test(image):
    # Calculate the total number of pixel pairs in horizontal and vertical directions
    num_horizontal_pairs = (image.shape[0] - 1) * image.shape[1]
    num_vertical_pairs = image.shape[0] * (image.shape[1] - 1)

    # Calculate the mean pixel value of the image
    mean_pixel_value = np.mean(image)

    # Initialize the correlation sums for horizontal and vertical directions
    correlation_horizontal = 0
    correlation_vertical = 0

    # Calculate the correlation for horizontal pixel pairs
    for i in range(image.shape[0]):
        for j in range(image.shape[1] - 1):
            correlation_horizontal += (image[i, j] - mean_pixel_value) * (image[i, j + 1] - mean_pixel_value)

    # Calculate the correlation for vertical pixel pairs
    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1]):
            correlation_vertical += (image[i, j] - mean_pixel_value) * (image[i + 1, j] - mean_pixel_value)

    # Normalize the correlation values by dividing by the total number of pixel pairs
    correlation_horizontal /= num_horizontal_pairs
    correlation_vertical /= num_vertical_pairs

    return correlation_horizontal, correlation_vertical


def grid_test(image, grid_size=(8, 8)):
    # Calculate the overall average pixel value of the image
    overall_mean = np.mean(image)

    # Calculate the size of each grid cell
    cell_height = image.shape[0] // grid_size[0]
    cell_width = image.shape[1] // grid_size[1]

    # Initialize the grid with zeros
    grid = np.zeros(grid_size)

    # Calculate the average pixel value for each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            grid[i, j] = np.mean(cell)

    # Calculate the difference between the grid cell averages and the overall average
    grid_difference = np.abs(grid - overall_mean)

    return grid_difference


def brightness_histogram(image):
    plt.hist(image.ravel(), 256, (0, 256))
    plt.show()


# Encrypt and decrypt the image using the previously defined sp_encrypt and sp_decrypt functions
encrypted_image = sp_encrypt(image, s_boxes, p_box)
decrypted_image = sp_decrypt(encrypted_image, s_boxes, p_box)

# Perform correlation tests
orig_corr_horizontal, orig_corr_vertical = correlation_test(image)
encr_corr_horizontal, encr_corr_vertical = correlation_test(encrypted_image)
decr_corr_horizontal, decr_corr_vertical = correlation_test(decrypted_image)

# Perform grid tests
grid_size = (8, 8)
orig_grid_difference = grid_test(image, grid_size)
encr_grid_difference = grid_test(encrypted_image, grid_size)
decr_grid_difference = grid_test(decrypted_image, grid_size)

# Display the original, encrypted, and decrypted images along with their brightness histograms,
# correlation test results, and grid test results
plt.figure(figsize=(18, 12))

# Original image
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title(f'Original Image\nHorizontal Corr: {orig_corr_horizontal:.2f}, Vertical Corr: {orig_corr_vertical:.2f}')

# Encrypted image
plt.subplot(3, 3, 2)
plt.imshow(encrypted_image, cmap='gray')
plt.title(f'Encrypted Image\nHorizontal Corr: {encr_corr_horizontal:.2f}, Vertical Corr: {encr_corr_vertical:.2f}')

# Decrypted image
plt.subplot(3, 3, 3)
plt.imshow(decrypted_image, cmap='gray')
plt.title(f'Decrypted Image\nHorizontal Corr: {decr_corr_horizontal:.2f}, Vertical Corr: {decr_corr_vertical:.2f}')

# Brightness histogram for the original image
plt.subplot(3, 3, 4)
plt.hist(image.ravel(), 256, (0, 256))
plt.title('Original Image Brightness Histogram')

# Brightness histogram for the encrypted image
plt.subplot(3, 3, 5)
plt.hist(encrypted_image.ravel(), 256, (0, 256))
plt.title('Encrypted Image Brightness Histogram')

# Brightness histogram for the decrypted image
plt.subplot(3, 3, 6)
plt.hist(decrypted_image.ravel(), 256, (0, 256))
plt.title('Decrypted Image Brightness Histogram')

# Grid test for the original image
plt.subplot(3, 3, 7)
plt.imshow(orig_grid_difference, cmap='hot', interpolation='nearest')
plt.title('Original Image Grid Test')

# Grid test for the encrypted image
plt.subplot(3, 3, 8)
plt.imshow(encr_grid_difference, cmap='hot', interpolation='nearest')
plt.title('Encrypted Image Grid Test')

# Grid test for the decrypted image
plt.subplot(3, 3, 9)
plt.imshow(decr_grid_difference, cmap='hot', interpolation='nearest')
plt.title('Decrypted Image Grid Test')

plt.tight_layout()
plt.show()
