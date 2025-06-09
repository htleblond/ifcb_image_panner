# ifcb_image_panner.py
# Version 1.01a
# By Holly LeBlond
# Fisheries and Oceans Canada
# June 6th, 2025

import numpy as np
import argparse
import os

from PIL import Image
from PIL import ImageOps
import scipy.ndimage as ndimage
import random

def process_image(fn, desired_size=224, invert=False, resize_scalar=1, c_size=5, use_var=False, rand_scalar=1.5, p_samples=10, gaussian_sigma=0):
    img = Image.open(fn)
    if invert:
        img = ImageOps.invert(img)
    width, height = img.size
    if resize_scalar != 1:
        img = img.resize((int(np.ceil(width*resize_scalar)),int(np.ceil(height*resize_scalar))))
        width, height = img.size
    img_list = list(img.getdata())
    img = []
    img_row = []
    for i in np.arange(len(img_list)):
        pixel = img_list[i]
        img_row.append(float(pixel))
        if np.mod(i+1, width) == 0:
            img.append(img_row)
            img_row = []
    c_size = np.min([c_size, width, height])
    tl_corner = [row[:c_size] for row in img[:c_size]]
    tr_corner = [row[-c_size:] for row in img[:c_size]]
    bl_corner = [row[:c_size] for row in img[-c_size:]]
    br_corner = [row[-c_size:] for row in img[-c_size:]]
    corners = [tl_corner, tr_corner, bl_corner, br_corner]
    background_median = np.median(corners)
    width_samples = [random.randint(0, np.abs(desired_size-width)) for x in np.arange(p_samples)]
    height_samples = [random.randint(0, np.abs(desired_size-height)) for x in np.arange(p_samples)]
    sample_pans = []
    if width > desired_size or height > desired_size:
        pan_pos = []
        if width > desired_size and height > desired_size:
            pan_pos = [(height_samples[x], width_samples[x]) for x in np.arange(p_samples)]
            sample_pans = [[row[coords[1]:coords[1]+desired_size] for row in img[coords[0]:coords[0]+desired_size]] for coords in pan_pos]
        elif width > desired_size:
            pan_pos = [(0, width_samples[x]) for x in np.arange(p_samples)]
            sample_pans = [[row[coords[1]:coords[1]+desired_size] for row in img] for coords in pan_pos]
        elif height > desired_size:
            pan_pos = [(height_samples[x], 0) for x in np.arange(p_samples)]
            sample_pans = [[row for row in img[coords[0]:coords[0]+desired_size]] for coords in pan_pos]
        sample_pans = [np.abs(np.array(x)-background_median) for x in sample_pans]
        sample_pans = [x.tolist() for x in sample_pans]
        foreground_means = [np.mean(x) for x in sample_pans]
        pan = pan_pos[np.argmax(foreground_means)]
        img = [row[pan[1]:np.min([pan[1]+desired_size, width])] for row in img[pan[0]:np.min([pan[0]+desired_size, height])]]
    if width < desired_size or height < desired_size:
        pan = (0, 0)
        if width < desired_size and height < desired_size:
            pan = (height_samples[0], width_samples[0])
        elif width < desired_size:
            pan = (0, width_samples[0])
        elif height < desired_size:
            pan = (height_samples[0], 0)
        empty = [[0 for x1 in np.arange(desired_size)] for x2 in np.arange(desired_size)]
        for i in np.arange(desired_size):
            for j in np.arange(desired_size):
                if pan[0] <= i < pan[0]+height and pan[1] <= j < pan[1]+width:
                    empty[i][j] = img[i-pan[0]][j-pan[1]]
                else:
                    if use_var:
                        empty[i][j] = background_median + np.floor(np.power(random.uniform(-1,1),3)*np.var(corners)*rand_scalar)
                    else:
                        empty[i][j] = background_median + np.floor(np.power(random.uniform(-1,1),3)*np.std(corners)*rand_scalar)
                    if empty[i][j] < 0:
                        empty[i][j] = 0
                    elif empty[i][j] > 255:
                        empty [i][j] = 255
        img = empty
    if gaussian_sigma != 0:
        img = ndimage.gaussian_filter(img, sigma=gaussian_sigma, order=0)
    return img

def main(args):
    ipn = args.input_path
    opn = args.output_path
    desired_size = args.desired_size
    invert = args.invert
    resize_scalar = args.resize_scalar
    c_size = args.c_size
    use_var = args.use_var
    rand_scalar = args.rand_scalar
    p_samples = args.p_samples
    gaussian_sigma = args.gaussian_sigma
    assert desired_size > 0, "desired_size must be an integer greater than 0"
    assert resize_scalar > 0, "resize_scalar must be greater than 0"
    assert c_size > 0, "c_size must be greater than 0"
    assert p_samples > 0, "p_samples must be greater than 0"
    if ipn[-1] != '/':
        ipn += '/'
    if opn[-1] != '/':
        opn += '/'
    classes = []
    classes_fns = {}
    for dn in os.listdir(ipn):
        if not os.path.isdir(ipn+dn):
            continue
        classes.append(dn)
        classes_fns[dn] = []
        for fn in os.listdir(ipn+dn):
            classes_fns[dn].append(fn)
    with open(opn+"settings_log.txt", "w") as f:
        f.write("input_path="+ipn+"\n")
        f.write("desired_size="+str(desired_size)+"\n")
        f.write("invert="+str(invert)+"\n")
        f.write("resize_scalar="+str(resize_scalar)+"\n")
        f.write("c_size="+str(c_size)+"\n")
        f.write("use_var="+str(use_var)+"\n")
        f.write("rand_scalar="+str(rand_scalar)+"\n")
        f.write("p_samples="+str(p_samples)+"\n")
        f.write("gaussian_sigma="+str(gaussian_sigma)+"\n")
    for key in classes:
        print("Processing "+str(len(classes_fns[key]))+" images from "+key+"...")
        os.makedirs(opn+key, exist_ok=True)
        for fn in classes_fns[key]:
            img = process_image(ipn+key+"/"+fn, desired_size=desired_size, invert=invert, resize_scalar=resize_scalar, \
                                c_size=c_size, use_var=use_var, rand_scalar=rand_scalar, p_samples=p_samples, \
                                gaussian_sigma=gaussian_sigma)
            img = Image.fromarray(np.array(img, dtype=np.uint8), mode='L')
            img.save(opn+key+"/"+fn)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, \
        help='Path to folder containing taxon-labelled subfolders with corresponding greyscale images for input.')
    parser.add_argument('output_path', type=str, \
        help='Path to folder that images will be output to. Subfolders are created automatically.')
    parser.add_argument('--desired_size', default=224, type=int, \
        help='The desired width and height of the output image. Note that the ifcb_classifier uses 224x224 images, which is the default here.')
    parser.add_argument('--invert', default=False, action='store_true', \
        help='Include this tag to invert the image.')
    parser.add_argument('--resize_scalar', default=1, type=float, \
        help='Images will be initially re-sized by multiplying their dimensions by the input number - more or less zooming in or out. \
                Default value is 1 (off).')
    parser.add_argument('--c_size', default=5, type=int, \
        help='Dimensions of boxes taken from the image corners used to calculate the background median and standard deviation values \
                for stochastic padding. Default value is 5.')
    parser.add_argument('--use_var', default=False, action='store_true', \
        help='Include this tag to multiply the random value for each pixel by the variance of the \'corner boxes\' instead of the standard \
                deviation for stochastic padding.')
    parser.add_argument('--rand_scalar', default=1.5, type=float, \
        help='The additional scalar the pixel values are multiplied by for stochastic padded. Default value is 1.5.')
    parser.add_argument('--p_samples', default=10, type=int, \
        help='The number of random coordinates generated to find the optimal panning placement. Default value is 10.')
    parser.add_argument('--gaussian_sigma', default=0, type=int, \
        help='The sigma value for optional Gaussian smoothing. Default value is 0 (off).')
    args = parser.parse_args()
    main(args)