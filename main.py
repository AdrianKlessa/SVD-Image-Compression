import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image

compression_factor = 0.1 # 0-1, inverted: 1 means using all singular values, 0 none


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
    return image


def get_svd(matrix_to_decompose):
    svd_result = np.linalg.svd(matrix_to_decompose, full_matrices=False)
    return(svd_result[0],svd_result[1],svd_result[2])

def multiply_svd(singular_values, U, S, V):
    #print(U.shape)
    #print(len(S))
    #print(V.shape)

    result = np.matmul(np.matmul(U[:,:singular_values], np.diag(S[:singular_values])), V[:singular_values,:])
    return result

def compress_and_reconstruct(R,G,B):
    result = []
    for element in (R,G,B):
        no_of_singular_values = math.ceil(max(R.shape) * compression_factor)
        print(f"Number of singular values: {no_of_singular_values}")
        U, S, V = get_svd(element)
        #print(f"U:{U}")
        #print(f"S:{S}")
        #print(f"V:{V}")
        result.append(multiply_svd(no_of_singular_values,U,S,V))
    return result


if __name__ == '__main__':
    image = load_image("test_images/curiosity.jpg")
    print(f"image shape when loaded: {image.shape}")

    # Split into 2d arrays for SVD
    R, G, B = [image[:, :, i] for i in range(3)]
    # TODO: Remove test code
    #print(R)
    #print(G)
    #print(B)

    #no_of_singular_values = math.ceil(max(R.shape)*compression_factor)
    #print(no_of_singular_values)
    #R_U, R_S, R_V = get_svd(R)
    #print(multiply_svd(no_of_singular_values,R_U,R_S,R_V))
    reconstructed_image = np.dstack(compress_and_reconstruct(R,G,B)).astype(np.uint8)
    #print(reconstructed_image.shape)
    #print(reconstructed_image[:,:,0])
    plt.imshow(reconstructed_image, interpolation="nearest")
    plt.show()
    matplotlib.image.imsave('compressed.png',reconstructed_image)

