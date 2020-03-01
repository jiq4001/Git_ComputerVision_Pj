import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations
from scipy.ndimage import gaussian_filter, sobel, convolve, maximum_filter, generic_filter
from cv2 import warpAffine
from cv2 import INTER_LINEAR

def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        dx = sobel(srcImage, axis = 1, mode = 'reflect')
        dy = sobel(srcImage, axis = 0, mode = 'reflect')
                
        xx = dx * dx
        yy = dy * dy
        xy = dx * dy
        
        wxx = gaussian_filter(xx, sigma = 0.5, mode='reflect')
        wyy = gaussian_filter(yy, sigma = 0.5, mode='reflect')
        wxy = gaussian_filter(xy, sigma = 0.5, mode='reflect')
        
        harrisImage = wxx*wyy - wxy*wxy - 0.1*(wxx+wyy)*(wxx+wyy)
        orientationImage = np.arctan2(dy, dx)*(180) / np.pi 
        # TODO-BLOCK-END

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        local_max_array = maximum_filter(harrisImage, size = 7, mode = 'reflect')
        destImage = (harrisImage == local_max_array)
        # TODO-BLOCK-END

        return destImage


    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.pt = (x, y)         
                f.size = 10
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                # TODO-BLOCK-END

                features.append(f)
        #return features
        
        # ANMS post processing
        # max number of features
        if (len(features)>1500):
            n = 1500+1
        else: 
            return features
        
        # global max, also first element of out_features[]
        sorted_key = sorted(features, key=lambda x: x.response, reverse=True)
        r = []
        out_features = []
        for i in range(len(sorted_key)):
            r_i = math.inf
            p_i = sorted_key[i]
            x = p_i.pt[0]
            y = p_i.pt[1]
            for j in range(len(sorted_key)):
                p_j = sorted_key[j]
                u = p_j.pt[0]
                v = p_j.pt[1]
                if (i !=j) and (p_i.response < (0.9 * p_j.response)):
                    d = (x-u)**2 + (y-v)**2
                else:
                    continue
                if (d < r_i):
                    r_i = d
            r.append(r_i)
        r = sorted(range(len(r)), key=lambda k: r[k], reverse = True)
        
        for i in r[:n]:
                out_features.append(sorted_key[i])
            
        return out_features
    
        


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        # import pdb; pdb.set_trace()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        desc = np.zeros((len(keypoints), 5 * 5))

        padimg = np.pad(grayImage, 2, 'constant', constant_values= 0)
        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            x = x+2
            y = y+2
            tem = padimg[y-2:y+3, x-2:x+3].reshape((1, 25))
            desc[i] = tem
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            x,y = f.pt

            arc = -f.angle * (np.pi) / 180
            

            trans_mx1 = np.array([[1,0,-x],
                                  [0,1,-y],
                                  [0,0,1]])


            rot_mx = np.array([  [math.cos(arc), -math.sin(arc), 0],
                                 [math.sin(arc), math.cos(arc), 0],
                                 [0, 0, 1]])
            

            scale_mx = np.array([[1/5,0,0],
                                 [0,1/5,0],
                                 [0,0,1]])

            trans_mx2 = np.array([[1,0,4], 
                                  [0,1,4], 
                                  [0,0,1]])


            
            transMx = np.dot(trans_mx2, np.dot(scale_mx, np.dot(rot_mx, trans_mx1)))[0:2,0:3]

            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR).reshape((1, windowSize * windowSize))

            # TODO 6: Normalize the descriptor to have zero mean and unit 
            # variance. If the variance is negligibly small (which we 
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            destImage = destImage - np.mean(destImage)
            dest_sd = np.std(destImage)
            if(dest_sd < 1e-5):
                destImage = np.zeros((1,windowSize * windowSize))
                
            else:
                destImage = destImage / dest_sd
            desc[i] = destImage
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



        
        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            x,y = f.pt
  
            arc = -f.angle * (np.pi) / 180
            

            trans_mx1 = np.array([[1,0,-x],
                                  [0,1,-y],
                                  [0,0,1]])


            rot_mx = np.array([  [math.cos(arc), -math.sin(arc), 0],
                                 [math.sin(arc), math.cos(arc), 0],
                                 [0, 0, 1]])
            

            scale_mx = np.array([[1,0,0],
                                [0,1,0],
                               [0,0,1]])

            trans_mx2 = np.array([[1,0,4], 
                                  [0,1,4], 
                                  [0,0,1]])
                            
            transMx = np.dot(trans_mx2, np.dot(scale_mx, np.dot(rot_mx, trans_mx1)))[0:2,0:3]
        
        
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage_stack = []
            dest_var = []
            for s in range(4):
                Sigma = 1.6
                grayImage_1 = ndimage.gaussian_filter(grayImage, Sigma* (np.sqrt(2))**(s-1))
                
                destImage = cv2.warpAffine(grayImage, transMx,
                       (windowSize, windowSize), flags=cv2.INTER_LINEAR).reshape((1, windowSize * windowSize))
                destImage = destImage - np.mean(destImage)
                
                destImage_stack.append(destImage)
                v = np.var(destImage)
                dest_var.append(v)
            destImage = destImage_stack[np.argmax(dest_var)]  
            
            # TODO 6: Normalize the descriptor to have zero mean and unit 
            # variance. If the variance is negligibly small (which we 
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            dest_sd = np.std(destImage)
            if(dest_sd < 1e-5):
                destImage = np.zeros((1,windowSize * windowSize))
                
            else:
                destImage = destImage / dest_sd
            desc[i] = destImage    

        return desc    

        
            
            
        
        


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        dis_mat = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')
        for queryIdx in range(desc1.shape[0]):
            trainIdx = np.argmin(dis_mat[queryIdx])
            distance = dis_mat[queryIdx, trainIdx]
            p = cv2.DMatch(queryIdx, trainIdx, distance)
            matches.append(p)         
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        dis_mat = scipy.spatial.distance.cdist(desc1, desc2, metric='euclidean')
        for queryIdx in range(desc1.shape[0]):
            sort_d = np.sort(dis_mat[queryIdx])
            trainIdx = np.argmin(dis_mat[queryIdx])
            first = sort_d[0]
            
            if len(sort_d) == 1:
                p = cv2.DMatch(queryIdx, trainIdx, 0)
                matches.append(p) 
                #return []
            else:
                second = sort_d[1]
                               
                if second == 0:
                    p = cv2.DMatch(queryIdx, trainIdx, 1)
                    matches.append(p)
                else:
                    distance = first / second
                    p = cv2.DMatch(queryIdx, trainIdx, distance)
                    matches.append(p)     
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
