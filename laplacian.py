### ------------------------------------------------------------------------- ###
#                                 Laplacian.py                                  #
### ------------------------------------------------------------------------- ###
#           Calculates the Laplacian of an input image using convolution        #
### ------------------------------------------------------------------------- ###
# All code adapted from:                                                        #
# Rosebrock, A. (2016). Convolutions with OpenCV and Python.                    #
# Retrieved from                                                                #
# https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/ #
### ------------------------------------------------------------------------- ###

# Import the packages that will be used in this code
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

# Define method for convolving an image (img) with a kernel (k)
def conv(img: np.ndarray, k: np.ndarray):
    """ A method for performing a 2D convolution between an Image (img) and a kernel (k) 
    
    This method takes two parameters, img, a np.array, and a kernel, k, another np.array.
    This method adds padded borders to the image according to the kernel size - 1 // 2,
    and then performs a convolution according to O(x,y)= ∑_(k=1)^m ∑_(l=1)^n I(x+k-1,y+l-1)×K(k,l),
    which is the elementwise multiplication of the image and kernel, summing the result for each
    intensity value (x,y). The resulting np.array is then rescaled to be in the range [0,255]
    and returned.

    Parameters
    ----------
    img : np.array
        The input image to convolve with the kernel

    k : np.array
        The kernel to convolve the image with
    """

    # Get the dimensions of the input image and kernel, in two dimensions
    (imgHeight, imgWidth) = img.shape[:2]
    (kHeight, kWidth) = k.shape[:2]

    # Need to create padding for the image so that the kernel will fit in the extremities of the image.
    # This is to produce an output image with the same dimensions as the input image.
    # For a square kernel (which the Laplacian is), this is the distance from the center of the kernel,
    # OR the width of the kernel - 1, then dividing by 2, flooring the answer to get an int. 
    # If for example the kernel is 3 x 3, i.e.
    # x x x
    # x x x
    # x x x
    # To fit this kernel in the top left corner of the image, we need to pad by 3 - 1 // 2 = 1
    # on each edge of the image
    padding = (kWidth - 1) // 2 

    # Copy the input img and add the padding to the edges.
    # Uses BORDER_REPLICATE to replicate the intensity at the edges of the image, instead of padding by 0s.
    # This is to retain consistency from the input to the output image.
    pad_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    # Initialise output image with the same dimensions as original image.
    # Using float 32 for incase the intensity number goes over 255 when convolving. 
    out = np.zeros((imgHeight, imgWidth), dtype="float32")

    # Now lets actually do the convolution

    # Loop over the input image for each (x,y) coordinate
    # The image actually starts at x = padding, y = padding, rather than 0. 
    # This is accounted for in the range.
    for y in np.arange(padding, imgHeight + padding):
        for x in np.arange(padding, imgWidth + padding):
            # Get the region of interest, i.e. the area in the image that the kernel will be applied to
            # This will get the region around the center of the kernel.
            # i.e. for a 3x3 kernel, where C is the center of the kernel and o are the surrounding intensities
            # gets a matrix as the following:
            # o  o  o
            # o  C  o
            # o  o  o
            # This is for x in the range [y - padding, y + padding + 1]
            # and y in the range [x - padding, x + padding + 1]
            region = pad_img[y - padding : y + padding + 1, x - padding : x + padding + 1]

            # Perform the convolution using elementwise multiplication of the region of interest matrix
            # and the kernel matrix, summing the result to give the intensity
            conv_intensity = (region * k).sum()

            # Store the convolved value in the output image at (x,y), removing the padding (there is no padding in the output img)
            out[y - padding, x - padding] = conv_intensity

    # Now since some of these intensities may be outside the range of [0,255], we need to rescale the 
    # intensities in the output image to be in the range [0,255]
    out = rescale_intensity(out, in_range=(0, 255))

    # Now that we have dealt with values over 255, we can convert our image intensities back to 8-bit
    out = (out * 255).astype("uint8")

    # Return our convolved image
    return out

def laplacian(img: np.ndarray, type: int):
    """ Method that calculates the Laplacian of the input image.

    This method takes two parameters, the image to calculate the Laplacian of, img, and an int 
    to select the type of Laplacian kernel to use. This is because a laplacian kernel has many
    types. All of which are 3x3 in this example. With 4 positive and negative peak, and 8 based
    positive and negative peak. The selected kernel is then convolved with the input image,
    and displayed in three imshow windows, one for the original input image, one for the output
    Laplacian, and one for the OpenCV version of the Laplacian. The method will wait for any
    key to be pressed to close the windows.

    Parameters
    ----------

    img : np.ndarray
        Image to calculate the Laplacian of.

    type : int
        Type of Laplacian to calculate.
    """

    print("Applying Laplacian kernel number {}".format(type))
    
    # Construct the Laplacian kernels. There are multiple types, as referenced from:
    # R. Fisher, S. Perkins, A. Walker and E. Wolfart. (2003),
    # Laplacian/Laplacian of Gaussian, URL: https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm, Accessed 29/05/2020
    laplacian_type = {
        0: np.array(([0, 1, 0],[1, -4, 1],[0, 1, 0]), dtype="int"),
        1: np.array(([0, -1, 0],[-1, 4, -1],[0, -1, 0]), dtype="int"),
        2: np.array(([1, 1, 1],[1, -8, 1],[1, 1, 1]), dtype="int"),
        3: np.array(([-1, -1, -1],[-1, 8, -1],[-1, -1, -1]), dtype="int")
    }

    # Convolve our input image with the laplacian filter
    convolve_output = conv(img, laplacian_type.get(type))

    # Show the original image
    cv2.imshow("Original", gray)

    # Show the image after applying the laplacian filter
    cv2.imshow("Custom Laplacian Filter", convolve_output)

    # Show the OpenCV version of the Laplacian too, for comparison.
    opencv_laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_32F, ksize=3))
    cv2.imshow("OpenCV Laplacian", opencv_laplacian)

    # Press any key to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Construct the argument parser for parsing image in command line.
# This allows any image to be used with --image *path*
ap = argparse.ArgumentParser(description="Compute Laplacian of input image")
ap.add_argument("-i", "--image", required=True, help="path to the input image")

# Add another argument for what laplacian filter kernel to use. Optional, defaults as 0. 
ap.add_argument("-t", "--type", nargs='?', const=0, default=0, required=False, help="optional, type of laplacian filter to use", type=int)
args = vars(ap.parse_args())

# Load the input image from the argument and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the laplacian type if it is defined
lap_type = args["type"]

# Calculate and display the laplacian
laplacian(gray, lap_type)