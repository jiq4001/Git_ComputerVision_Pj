import sys
import cv2
import numpy as np


def pad_img(img, kernel):
    if type(kernel) == int:
        return(img) 
    elif (kernel.shape[0] != 1) & (kernel.shape[1] != 1):
        pad_x = img.shape[0] + (kernel.shape[0] - 1) #padding x
        pad_y = img.shape[1] + (kernel.shape[1] - 1) #padding y
        padding = ((kernel.shape[0] - 1)/2, (kernel.shape[1] - 1)/2) 
    elif (kernel.shape[0] == 1) & (kernel.shape[1] != 1):
        pad_x = img.shape[0] #padding x
        pad_y = img.shape[1] + (kernel.shape[1] - 1) #padding y
        padding = (0, (kernel.shape[1] - 1)/2)    
    else: 
        pad_x = img.shape[0] + (kernel.shape[0] - 1) #padding x
        pad_y = img.shape[1] #padding y
        padding = ((kernel.shape[0] - 1)/2, 0)    

                
    if img.ndim == 3:
        padded_img = np.zeros((pad_x, pad_y, 3))
        for i in range(3):
            tem = padded_img[:,:,i]
            replace = tuple([slice(padding[dim], padding[dim] + img.shape[dim]) for dim in range(img.ndim-1)])
            tem[replace] = img[:,:,i]
            padded_img[:,:,i]=tem
    else:
        padded_img = np.zeros((pad_x, pad_y))
        replace = tuple([slice(padding[dim], padding[dim] + img.shape[dim]) for dim in range(img.ndim)])
        padded_img[replace] = img  
    return(padded_img)


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    padded_img = pad_img(img, kernel)
    if type(kernel) == int:
        out = img
    else:
        if padded_img.ndim == 3:
            out = np.zeros((img.shape[0], img.shape[1], 3))
            for i in range(3):
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        for j in range(kernel.shape[0]):
                            for k in range(kernel.shape[1]):
                                out[x, y, i] += padded_img[x+j, y+k, i] * kernel[j, k]

        else:
            out = np.zeros((img.shape[0], img.shape[1]))
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    for j in range(kernel.shape[0]):
                        for k in range(kernel.shape[1]):
                            out[x, y] += padded_img[x+j, y+k] * kernel[j, k]

        return(out)

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    padded_img = pad_img(img, kernel)
    if type(kernel) == int:
        out = img
    else:
        if padded_img.ndim == 3:
            out = np.zeros((img.shape[0], img.shape[1], 3))
            for i in range(3):
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        for j in range(kernel.shape[0]):
                            for k in range(kernel.shape[1]):
                                out[x, y, i] += padded_img[x+j, y+k, i] * kernel[kernel.shape[0]-j-1, kernel.shape[1]-k-1]

        else:
            out = np.zeros((img.shape[0], img.shape[1]))
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    for j in range(kernel.shape[0]):
                        for k in range(kernel.shape[1]):
                            out[x, y] += padded_img[x+j, y+k] * kernel[kernel.shape[0]-j-1, kernel.shape[1]-k-1]

        return(out)

    
def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    if (height != 1) & (width != 1):
        h, w = np.meshgrid(np.linspace(-(width-1)/2,(width-1)/2,width), np.linspace(-(height-1)/2,(height-1)/2,height))
        d = np.sqrt(w * w + h * h)
        g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ))) *  (1/(2* np.pi*sigma**2))
        g = g/np.sum(g)
    elif (height ==1) & (width != 1):
        w = np.linspace(-(width-1)/2,(width-1)/2,width)
        d = np.sqrt(w * w)    
        g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ))) * 1/(np.sqrt(2*np.pi) * sigma)
        g = g/np.sum(g)   
    elif (height !=1) & (width == 1):
        h = np.linspace(-(height-1)/2,(height-1)/2,height)
        d = np.sqrt(h * h)
        g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ))) * 1/(np.sqrt(2*np.pi) * sigma)
        g = g/np.sum(g)        
    else:
        g = 1
    return(g)

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    if size == 1:
        out = img 
    else:
        #filter_kernel = np.ones(size**2).reshape(size, size)/sigma ### mean filter
        filter_kernel = gaussian_blur_kernel_2d(sigma, size, size)
        out = convolve_2d(img, filter_kernel) 
    return(out) 

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    if size == 1:
        out = img 
    else:
        #filter_kernel = np.ones(size**2).reshape(size, size)/sigma ### mean filter
        filter_kernel = gaussian_blur_kernel_2d(sigma, size, size)
        low = convolve_2d(img, filter_kernel)
        out = np.subtract(img, low)
    return(out)  

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

