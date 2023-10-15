import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image

compression_factor = 0.1  # 0-1, inverted: 1 means using all singular values, 0 none


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
    return image


def get_svd(matrix_to_decompose):
    svd_result = np.linalg.svd(matrix_to_decompose, full_matrices=False)
    return (svd_result[0], svd_result[1], svd_result[2])


def multiply_svd_compress(singular_values, U, S, V):
    # print(U.shape)
    # print(len(S))
    # print(V.shape)

    result = np.matmul(np.matmul(U[:, :singular_values], np.diag(S[:singular_values])), V[:singular_values, :])
    return result


def get_svd_compressed(singular_values, matrix_to_decompose):
    (U, S, V) = np.linalg.svd(matrix_to_decompose, full_matrices=False)
    U_c = U[:, :singular_values]
    S_c = S[:singular_values]
    V_c = V[:singular_values, :]
    return (U_c, S_c, V_c)


def multiply_compressed(U, S, V):
    return np.matmul(np.matmul(U, np.diag(S)), V)


def compress_and_reconstruct(R, G, B):
    result = []
    for element in (R, G, B):
        no_of_singular_values = math.ceil(max(R.shape) * compression_factor)
        print(f"Number of singular values: {no_of_singular_values}")
        U, S, V = get_svd(element)
        # print(f"U:{U}")
        # print(f"S:{S}")
        # print(f"V:{V}")
        result.append(multiply_svd_compress(no_of_singular_values, U, S, V))
    return result


def reconstruct(compressed_image):
    result = []
    for element in compressed_image:
        U, S, V = element
        result.append(multiply_compressed(U, S, V))
    return np.dstack(result).astype(np.uint8)


def get_compressed_image_size(compressed_image):
    result = 0
    counter = 0
    for dim in compressed_image:
        for element in dim:
            result += element.size
            counter += 1
    print(f"Went through {counter} dimensions")
    return result


if __name__ == '__main__':
    image = load_image("test_images/curiosity.jpg")
    print(f"image shape when loaded: {image.shape}")
    print(f"Loaded image size in bytes: {image.size}")

    # Split into 2d arrays for SVD
    R, G, B = [image[:, :, i] for i in range(3)]
    # TODO: Remove test code
    no_of_singular_values = math.ceil(max(R.shape) * compression_factor)
    compressed_image = [get_svd_compressed(no_of_singular_values, dim) for dim in (R, G, B)]
    compressed_size = get_compressed_image_size(compressed_image)
    reconstructed_image = reconstruct(compressed_image)

    print(f"Original image size: {image.size}")
    print(f"Compressed image size: {compressed_size}")
    space_saved = image.size - compressed_size
    space_saved_percent = (1 - (compressed_size / image.size)) * 100
    print(f"Space saved in bytes: {image.size - compressed_size} ({(1 - (compressed_size / image.size)) * 100:.2f}%)")
    plt.imshow(reconstructed_image, interpolation="nearest")
    plt.show()
    matplotlib.image.imsave(f'Compressed by {space_saved_percent:.2f} percent.png', reconstructed_image)
