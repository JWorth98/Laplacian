# Laplacian

Calculates the Laplacian of an input image.

## Installation
Clone project to local machine. Install a python interpreter. Run the application, as detailed in Usage.

## Usage

Run from python command prompt in the project folder with valid python interpreter.

A test version of the program can be used with the image blurry_moon.tif
```
usage: python laplacian.py [-h] -i IMAGE [-t [TYPE]]

Compute Laplacian of input image

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -t [TYPE], --type [TYPE]
                        optional, type of laplacian filter to use
                        0 (default) = [ 0, 1, 0]
                                      [ 1,-4, 1]
                                      [ 0, 1, 0]

                        1 =           [ 0,-1, 0]
                                      [-1, 4,-1]
                                      [ 0,-1, 0]

                        2 =           [ 1, 1, 1]
                                      [ 1,-8, 1]
                                      [ 1, 1, 1]

                        3 =           [-1,-1,-1]
                                      [-1, 8,-1]
                                      [-1,-1,-1]

```

## Required Packages
- OpenCV
- Numpy
- scikit-image
- argparse

## Acknowledgements
Thank you to Rosebrock, A (2016) for providing sample code. All code was adapted from the tutorial found at `https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/`.
