# SVD-Image-Compression

A simple example of using Singular Value Decomposition for image compression. The example below is compressed by 76% (from a (710, 946, 3) uint8 numpy array to 9 2D arrays using the float64 datatype). 

(2MB to 472 KB)

Original:

![alt text](test_images/curiosity.jpg)

Compressed:

![alt text](https://github.com/AdrianKlessa/SVD-Image-Compression/blob/b225001415aba2bbfe36c35aaa2e1841d682468c/Compressed%20by%2076.56%20percent.png)

Compression rate is controlled with the `compression_factor` global variable, determining the percentage of singular values to be used.

More compression would likely be achieved with minimal quality loss by forcing usage of lower-precision datatypes than the default float64. 
