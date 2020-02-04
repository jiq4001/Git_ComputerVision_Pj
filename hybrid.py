import sys
import cv2
import numpy as np



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
        replace = [slice(padding[dim], padding[dim] + img.shape[dim]) for dim in range(img.ndim)]
        padded_img[replace] = img  
    return(padded_img)

##################

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
    padded_img = cross_correlation_2d(img, kernel)
    
    if type(kernel) == int:
        out = img
    else:
        if img.shape[0] % kernel.shape[0] != 0:
            cov_shape_x = padded_img.shape[0] - kernel.shape[0] + 1
        else:
            cov_shape_x = img.shape[0]
        if img.shape[1] % kernel.shape[1] != 0:
            cov_shape_y = padded_img.shape[1] - kernel.shape[1] + 1
        else:
            cov_shape_y = img.shape[1]    
        
        cov_shape = (cov_shape_x, cov_shape_y) + kernel.shape
        strides = padded_img.strides[:2] *2
    
        if padded_img.ndim == 3:
            out = np.zeros((cov_shape_x, cov_shape_y, 3))
            for i in range(3):
                cov_mat = np.lib.stride_tricks.as_strided(padded_img[:,:,i], cov_shape, strides)  * np.flip(kernel) #kernel  
                for x in range(cov_mat.shape[0]):
                    for y in range(cov_mat.shape[1]):
                        out[x, y, i] = np.sum(cov_mat[x, y, :, :])
#            for x in range(cov_shape_x):
#                for y in range(cov_shape_y):
#                    tlum = np.sum(out[x,y, :])
#                    for i in range(3):
#                        out[x,y, i] = out[x,y, i]/ tlum 
            for i in range(3):
                maxlum=np.max(out[:,:, i])
                minlum=np.min(out[:,:, i])
                for x in range(cov_shape_x):
                    for y in range(cov_shape_y):
                        out[x,y, i] = (out[x,y, i] - minlum) * 255 / (maxlum - minlum)

        else:
            out = np.zeros((cov_shape_x, cov_shape_y))
            cov_mat = np.lib.stride_tricks.as_strided(padded_img, cov_shape, strides)  * np.flip(kernel) #kernel  
            for x in range(cov_mat.shape[0]):
                for y in range(cov_mat.shape[1]):
                    out[x, y] = np.sum(cov_mat[x, y, :, :]) 

    return(out)

##############

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
    mu = 0.0
    w, h = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
    if (height != 1) & (width != 1):
        d = np.sqrt(w * w + h * h)
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ))) *  (1/np.sqrt(2* np.pi*sigma**2))
        g = g/np.abs(g.max())
    elif (height ==1) & (width != 1):
        d = np.sqrt(w * w)    
        g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ))) * 1/(np.sqrt(2*np.pi) * sigma)
        g = g/np.abs(g.max())    
    elif (height !=1) & (width == 1):
        d = np.sqrt(h * h)
        g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ))) * 1/(np.sqrt(2*np.pi) * sigma)
        g = g/np.abs(g.max())        
    else:
        g = 1
    return(g)

##################

def low_pass(img, sigma, size):
    if size == 1:
        out = img / sigma
    else:
        #filter_kernel = np.ones(size**2).reshape(size, size)/sigma ### mean filter
        filter_kernel = gaussian_blur_kernel_2d(sigma, size, size)
        padded_img = cross_correlation_2d(img, filter_kernel)
    
        if img.shape[0] % filter_kernel.shape[0] != 0:
            cov_shape_x = padded_img.shape[0] - filter_kernel.shape[0] + 1
        else:
            cov_shape_x = img.shape[0]
        if img.shape[1] % filter_kernel.shape[1] != 0:
            cov_shape_y = padded_img.shape[1] - filter_kernel.shape[1] + 1
        else:
            cov_shape_y = img.shape[1]
        
        cov_shape = (cov_shape_x, cov_shape_y) + filter_kernel.shape
        strides = padded_img.strides[:2] *2
    
        if padded_img.ndim == 3:
            out = np.zeros((cov_shape_x, cov_shape_y, 3))
            for i in range(3):
                cov_mat = np.lib.stride_tricks.as_strided(padded_img[:,:,i], cov_shape, strides)  * np.flip(filter_kernel) #kernel  
                for x in range(cov_mat.shape[0]):
                    for y in range(cov_mat.shape[1]):
                        out[x, y, i] = np.sum(cov_mat[x, y, :, :])

        else:
            out = np.zeros((cov_shape_x, cov_shape_y))
            cov_mat = np.lib.stride_tricks.as_strided(padded_img, cov_shape, strides)  * np.flip(filter_kernel) #kernel  
            for x in range(cov_mat.shape[0]):
                for y in range(cov_mat.shape[1]):
                    out[x, y] = np.sum(cov_mat[x, y, :, :]) 

    return(out*0.005)   

        

def high_pass(img, sigma, size):
    if size == 1:
        out = img * (1 - 1/sigma)
    else:
        #filter_kernel = np.ones(size**2).reshape(size, size)/sigma ### mean filter
        filter_kernel = gaussian_blur_kernel_2d(sigma, size, size)
        padded_img = cross_correlation_2d(img, filter_kernel)
    
        if img.shape[0] % filter_kernel.shape[0] != 0:
            cov_shape_x = padded_img.shape[0] - filter_kernel.shape[0] + 1
        else:
            cov_shape_x = img.shape[0]
        if img.shape[1] % filter_kernel.shape[1] != 0:
            cov_shape_y = padded_img.shape[1] - filter_kernel.shape[1] + 1
        else:
            cov_shape_y = img.shape[1]
        
        cov_shape = (cov_shape_x, cov_shape_y) + filter_kernel.shape
        strides = padded_img.strides[:2] *2
    
        if padded_img.ndim == 3:
            out = np.zeros((cov_shape_x, cov_shape_y, 3))
            for i in range(3):
                cov_mat = np.lib.stride_tricks.as_strided(padded_img[:,:,i], cov_shape, strides)  * np.flip(filter_kernel) #kernel  
                for x in range(cov_mat.shape[0]):
                    for y in range(cov_mat.shape[1]):
                        out[x, y, i] = img[x, y, i] - np.sum(cov_mat[x, y, :, :])

        else:
            out = np.zeros((cov_shape_x, cov_shape_y))
            cov_mat = np.lib.stride_tricks.as_strided(padded_img, cov_shape, strides)  * np.flip(filter_kernel) #kernel  
            for x in range(cov_mat.shape[0]):
                for y in range(cov_mat.shape[1]):
                    out[x, y] = img[x, y] - np.sum(cov_mat[x, y, :, :]) 

        return(out*0.005)             
  

#################

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

