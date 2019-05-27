import sys
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib as m
from skimage import io, color
from skimage.io import imread, imsave
imwrite = imsave

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def initialize():
    if len(sys.argv) != 4:
        print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
        print("")
        print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
        print("If there's at least one single full-black pixel in ANNO, black is assumed to mean unknown.")
        sys.exit(1)

    fn_im = sys.argv[1]
    fn_anno = sys.argv[2]
    fn_output = sys.argv[3]
    RGB = rgb(fn_im,fn_anno).T
    HSV = hsv(fn_im,fn_anno).T
    LAB = lab(fn_im,fn_anno).T
    arr=np.array([RGB,HSV,LAB])
    print(arr.shape)
    prob = routing(arr,3)
    colorize(fn_im,fn_anno,fn_output,prob)


def rgb(fn_im,fn_anno):
    
    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)
    anno_rgb = imread(fn_anno).astype(np.uint32)
    #print(anno_rgb.shape)   #Dhriti
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
    #    print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
     #   print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    #colorize = np.empty((len(colors), 3), np.uint8)
    #colorize[:,0] = (colors & 0x0000FF)
    #colorize[:,1] = (colors & 0x00FF00) >> 8
    #colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)
    RGB = np.array(Q)
    return RGB


def hsv(fn_im,fn_anno):
    
    ##################################
    ### Read images and annotation ###
    ##################################
    img_rgb = imread(fn_im)
    img = m.colors.rgb_to_hsv(img_rgb)
    anno_rgb = imread(fn_anno).astype(np.uint32)
    #print(anno_rgb.shape)   #Dhriti
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
    #    print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
     #   print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    #colorize = np.empty((len(colors), 3), np.uint8)
    #colorize[:,0] = (colors & 0x0000FF)
    #colorize[:,1] = (colors & 0x00FF00) >> 8
    #colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)
    HSV = np.array(Q)
    return HSV


def lab(fn_im,fn_anno):
    ##################################
    ### Read images and annotation ###
    ##################################
    img_rgb = imread(fn_im)
    img = color.rgb2lab(img_rgb)
    anno_rgb = imread(fn_anno).astype(np.uint32)
    #print(anno_rgb.shape)   #Dhriti
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
    #    print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
     #   print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    #colorize = np.empty((len(colors), 3), np.uint8)
    #colorize[:,0] = (colors & 0x0000FF)
    #colorize[:,1] = (colors & 0x00FF00) >> 8
    #colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)
    LAB = np.array(Q)
    return LAB


#For calculating softmax of vectors
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


# numpy implementation of the probabilities
def _squash(vector):
    # squashing the given vector in numpy
    vector_squared_norm = np.sum(np.square(vector), keepdims=True)
    scalar_factor = vector_squared_norm / ( ( 1 + vector_squared_norm ) * np.sqrt(vector_squared_norm) )
    vector_squash = scalar_factor * vector
    return vector_squash


def colorize(fn_im,fn_anno,fn_output,prob):
    img = imread(fn_im)
    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread(fn_anno).astype(np.uint32)
    #print(anno_rgb.shape)   #Dhriti
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
    #    print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
     #   print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    print(colorize.shape)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    MAP = np.argmax(prob.T, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    imwrite(fn_output, MAP.reshape(img.shape))

#For routing of the probabilities
def routing(arr,it):
    #path1="/home/Dhriti/Music/bij.csv"
    #path2="/home/Dhriti/Music/ci.csv"
    #path3="/home/Dhriti/Music/vj.csv"
    Bij=np.zeros((arr.shape[1],3),dtype=float)
    Vj=np.zeros((arr.shape[1],arr.shape[2]),dtype=float)
    #print(bij.shape)
    for r in range(0,it):
        #print("For r ="+str(r))
        Ci=softmax(Bij,axis=1)
        #print("Ci")
        #print(Ci)
        #np.savetxt(path2, Ci, fmt='%.8f', delimiter=',')
        Sj=np.zeros((arr.shape[1],arr.shape[2]))
        for j in range(0,3):
            for i in range(0,arr.shape[1]):
                Sj[i] += arr[j][i]*Ci[i][j]
        #print("sj.shape")
        #print("sj")
        #print(sj.shape) 
        #print(Sj)
        for i in range(0,arr.shape[1]):
            Vj[i] = _squash(Sj[i])
            #print(Vj.shape)
        #print("vj")
        #print(Vj)
        #np.savetxt(path3, Vj, fmt='%.8f', delimiter=',')
        #for converting the tensors into numpy array
        #Vj=tf.Session().run(vj)
        #print(Vj)
        #print(Vj.shape)
        for j in range(0,3):
            for i in range(0,arr.shape[2]):
                Bij[i][j] = Bij[i][j] + np.dot(Vj[i],arr[j][i])
        #print("bij")
        #print(Bij)
        #np.savetxt(path1, Bij, fmt='%.8f', delimiter=',')
    #path1="/home/Dhriti/Videos/MSRC_routing.csv"
        #np.savetxt(path1, bij, fmt='%.8f', delimiter=',')
        #print(bij)
    #return Vj    
    #print(vj)
    #np.savetxt(path1, Vj, fmt='%.9f', delimiter=',')
    return Vj
    


def main():
    initialize()


if __name__=='__main__':
    main() 

