# ifcb_image_panner.py
# By Holly LeBlond
# Fisheries and Oceans Canada

import numpy as np
import argparse
import os

from PIL import Image
from PIL import ImageOps
import scipy.ndimage as ndimage
import random

from importlib.metadata import version
#from line_profiler import profile

version = '1.02a'

desired_size=299
best_only=False
hop_scalar=0.8
min_dim_scalar=0.35
invert=False
resize_scalar=1
c_size=5
use_var=False
rand_scalar=1.5
p_samples=5
gaussian_sigma=0
std_threshold=20.0
best_must_pass=False
verbose_off=False

ds_arange = np.arange(desired_size)

#@profile
def process_image(fn, desired_size=desired_size, best_only=best_only, hop_scalar=hop_scalar, min_dim_scalar=min_dim_scalar, invert=invert, \
                  resize_scalar=resize_scalar, c_size=c_size, use_var=use_var, rand_scalar=rand_scalar, p_samples=p_samples, \
                  gaussian_sigma=gaussian_sigma, std_threshold=std_threshold, best_must_pass=best_must_pass, verbose_off=verbose_off):
    orig_img = Image.open(fn)
    if invert:
        orig_img = ImageOps.invert(orig_img)
    width, height = orig_img.size
    if resize_scalar != 1:
        orig_img = orig_img.resize((int(np.ceil(width*resize_scalar)),int(np.ceil(height*resize_scalar))))
        width, height = orig_img.size
    orig_img_list = list(orig_img.getdata())
    orig_img = []
    orig_img_row = []
    for i in np.arange(len(orig_img_list)):
        pixel = orig_img_list[i]
        orig_img_row.append(float(pixel))
        if np.mod(i+1, width) == 0:
            orig_img.append(orig_img_row)
            orig_img_row = []
    c_size = np.min([c_size, width, height])
    tl_corner = [row[:c_size] for row in orig_img[:c_size]]
    tr_corner = [row[-c_size:] for row in orig_img[:c_size]]
    bl_corner = [row[:c_size] for row in orig_img[-c_size:]]
    br_corner = [row[-c_size:] for row in orig_img[-c_size:]]
    corners = [tl_corner, tr_corner, bl_corner, br_corner]
    background_median = np.median(corners)
    padding_scalar = rand_scalar
    if use_var:
        padding_scalar *= np.var(corners)
    else:
        padding_scalar *= np.std(corners)
    width_samples = [random.randint(0, np.abs(desired_size-width)) for x in np.arange(p_samples)]
    height_samples = [random.randint(0, np.abs(desired_size-height)) for x in np.arange(p_samples)]
    sample_pans = []
    imgs = []
    best_img = ([0.0], "x") # placeholder
    if width > desired_size or height > desired_size:
        pan_pos = []
        if width > desired_size and height > desired_size:
            pan_pos = [(height_samples[x], width_samples[x]) for x in np.arange(p_samples)]
            sample_pans = [[row[coords[1]:coords[1]+desired_size] for row in orig_img[coords[0]:coords[0]+desired_size]] for coords in pan_pos]
        elif width > desired_size:
            pan_pos = [(0, width_samples[x]) for x in np.arange(p_samples)]
            sample_pans = [[row[coords[1]:coords[1]+desired_size] for row in orig_img] for coords in pan_pos]
        elif height > desired_size:
            pan_pos = [(height_samples[x], 0) for x in np.arange(p_samples)]
            sample_pans = [[row for row in orig_img[coords[0]:coords[0]+desired_size]] for coords in pan_pos]
        sample_pans = [np.array(x)-background_median for x in sample_pans]
        sample_pans = [x.tolist() for x in sample_pans]
        foreground_stds = [np.std(x) for x in sample_pans]
        best_pan = pan_pos[np.argmax(foreground_stds)]
        tl_pan = [best_pan[0], best_pan[1]]
        while tl_pan[0] > 0:
            tl_pan[0] -= np.ceil(hop_scalar*desired_size)
        while tl_pan[1] > 0:
            tl_pan[1] -= np.ceil(hop_scalar*desired_size)
        hopped_pans = []
        for i in np.arange(np.ceil(height/(hop_scalar*desired_size))):
            for j in np.arange(np.ceil(width/(hop_scalar*desired_size))):
                hopped_pans.append((tl_pan[0]+np.ceil(hop_scalar*desired_size)*i, tl_pan[1]+np.ceil(hop_scalar*desired_size)*j))
        for pan in hopped_pans:
            if pan[0]+desired_size < 0 or pan[0] >= height or pan[1]+desired_size < 0 or pan[0] >= width:
                continue
            piece_type = "c" # centre
            if desired_size > width and desired_size <= height and pan[1] <= 0 and pan[1]+desired_size > width:
                if pan[0] < 0:
                    piece_type = "te" # top end
                elif pan[0]+desired_size >= height:
                    piece_type = "be" # bottom end
                else:
                    piece_type = "vc" # vertical centre
            elif desired_size > height and desired_size <= width and pan[0] <= 0 and pan[0]+desired_size > height:
                if pan[1] < 0:
                    piece_type = "le" # left end
                elif pan[1]+desired_size >= width:
                    piece_type = "re" # right end
                else:
                    piece_type = "hc" # horizontal centre
            else:
                if pan[0] < 0:
                    if pan[1] < 0:
                        piece_type = "tl" # top-left corner
                    elif pan[1]+desired_size >= width:
                        piece_type = "tr" # top-right corner
                    else:
                        piece_type = "t" # top edge
                elif pan[0]+desired_size >= height:
                    if pan[1] < 0:
                        piece_type = "bl" # bottom-left corner
                    elif pan[1]+desired_size >= width:
                        piece_type = "br" # bottom-right corner
                    else:
                        piece_type = "b" # bottom edge
                else:
                    if pan[1] < 0:
                        piece_type = "l" # left edge
                    elif pan[1]+desired_size >= width:
                        piece_type = "r" # right edge
            crop = [row[int(np.max([pan[1],0])):int(np.min([pan[1]+desired_size,width]))] \
                    for row in orig_img[int(np.max([pan[0],0])):int(np.min([pan[0]+desired_size,height]))]]
            imgs.append((crop, piece_type))
    else:
        best_img = (orig_img, "s") # small image
        imgs.append(best_img)
    outp_imgs = []
    best_img_contrast = np.std(np.array(best_img[0])-background_median)
    for img_data in imgs:
        img = img_data[0]
        piece_type = img_data[1]
        img_contrast = np.std(np.array(img)-background_median)
        #print(str(np.shape(img))+" "+piece_type)
        #print(img_contrast)
        if img_contrast > best_img_contrast:
            best_img = img_data
            best_img_contrast = img_contrast
        if img_contrast >= std_threshold and \
                ((piece_type in ["le", "re", "hc"] or len(img) >= desired_size*min_dim_scalar) and \
                 (piece_type in ["te", "be", "vc"] or len(img[0]) >= desired_size*min_dim_scalar)):
            if best_only:
                outp_imgs = [best_img]
            else:
                outp_imgs.append(img_data)
    if not best_must_pass and len(outp_imgs) == 0 and best_img[1] != 'x':
        outp_imgs.append(best_img)
    for i in np.arange(len(outp_imgs)):
        img_data = outp_imgs[i]
        img = img_data[0]
        piece_type = img_data[1]
        pan = (0, 0)
        if piece_type == "s":
            pan = (random.randint(0, desired_size-len(img)), random.randint(0, desired_size-len(img[0])))
        elif piece_type == "te":
            pan = (desired_size-len(img), random.randint(0, desired_size-len(img[0])))
        elif piece_type in ["be", "vc"]:
            pan = (0, random.randint(0, desired_size-len(img[0])))
        elif piece_type == "le":
            pan = (random.randint(0, desired_size-len(img)), desired_size-len(img[0]))
        elif piece_type in ["re", "hc"]:
            pan = (random.randint(0, desired_size-len(img)), 0)
        elif piece_type == "tl":
            pan = (desired_size-len(img), desired_size-len(img[0]))
        elif piece_type in ["tr", "t"]:
            pan = (desired_size-len(img), 0)
        elif piece_type in ["bl", "l"]:
            pan = (0, desired_size-len(img[0]))
        elif piece_type in ["br", "b", "r"]:
            pan = (0, 0)
        else: # c (centre piece) - no padding needed
            continue
        padded = [[0 for x1 in ds_arange] for x2 in ds_arange]
        for j in ds_arange:
            for k in ds_arange:
                if pan[0] <= j < pan[0]+len(img) and pan[1] <= k < pan[1]+len(img[0]):
                    padded[j][k] = img[j-pan[0]][k-pan[1]]
                else:
                    if use_var:
                        padded[j][k] = background_median + np.floor(np.power(random.uniform(-1,1),3)*padding_scalar)
                    else:
                        padded[j][k] = background_median + np.floor(np.power(random.uniform(-1,1),3)*padding_scalar)
                    if padded[j][k] < 0:
                        padded[j][k] = 0
                    elif padded[j][k] > 255:
                        padded [j][k] = 255
        if gaussian_sigma != 0:
            padded = ndimage.gaussian_filter(padded, sigma=gaussian_sigma, order=0)
        outp_imgs[i] = (padded, piece_type)
    return outp_imgs

def main(args):
    ipn = args.input_path
    opn = args.output_path
    desired_size = args.desired_size
    best_only = args.best_only
    hop_scalar = args.hop_scalar
    min_dim_scalar = args.min_dim_scalar
    invert = args.invert
    resize_scalar = args.resize_scalar
    c_size = args.c_size
    use_var = args.use_var
    rand_scalar = args.rand_scalar
    p_samples = args.p_samples
    gaussian_sigma = args.gaussian_sigma
    std_threshold = args.std_threshold
    best_must_pass = args.best_must_pass
    ds_arange = np.arange(desired_size)
    assert desired_size > 0, "desired_size must be an integer greater than 0"
    assert resize_scalar > 0, "resize_scalar must be greater than 0"
    assert c_size > 0, "c_size must be greater than 0"
    assert p_samples > 0, "p_samples must be greater than 0"
    assert hop_scalar > 0, "hop_scalar must be greater than 0"
    assert min_dim_scalar <= 1, "min_dim_scalar must be less than or equal to 1"
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
        f.write("best_only="+str(best_only)+"\n")
        f.write("hop_scalar="+str(hop_scalar)+"\n")
        f.write("min_dim_scalar="+str(min_dim_scalar)+"\n")
        f.write("invert="+str(invert)+"\n")
        f.write("resize_scalar="+str(resize_scalar)+"\n")
        f.write("c_size="+str(c_size)+"\n")
        f.write("use_var="+str(use_var)+"\n")
        f.write("rand_scalar="+str(rand_scalar)+"\n")
        f.write("p_samples="+str(p_samples)+"\n")
        f.write("gaussian_sigma="+str(gaussian_sigma)+"\n")
        f.write("std_threshold="+str(std_threshold)+"\n")
        f.write("best_must_pass="+str(best_must_pass)+"\n")
    for key in classes:
        if not verbose_off:
            print("Processing "+str(len(classes_fns[key]))+" images from "+key+"...")
        os.makedirs(opn+key, exist_ok=True)
        for fn in classes_fns[key]:
            outp_imgs = process_image(ipn+key+"/"+fn, desired_size=desired_size, best_only=best_only, \
                                hop_scalar=hop_scalar, min_dim_scalar=min_dim_scalar, invert=invert, resize_scalar=resize_scalar, \
                                c_size=c_size, use_var=use_var, rand_scalar=rand_scalar, p_samples=p_samples, \
                                gaussian_sigma=gaussian_sigma, std_threshold=std_threshold, best_must_pass=best_must_pass)
            piece_dict = {}
            for i in np.arange(len(outp_imgs)):
                img = outp_imgs[i][0]
                piece_type = outp_imgs[i][1]
                if piece_type not in piece_dict:
                    piece_dict[piece_type] = 0
                piece_dict[piece_type] += 1
                img = Image.fromarray(np.array(img, dtype=np.uint8), mode='L')
                img.save(opn+key+"/"+fn.split(".")[0]+"_"+str(i+1).zfill(3)+"_"+piece_type+".png")
            if not verbose_off:
                print(str(len(outp_imgs))+" image(s) created from "+key+"/"+fn+" "+str(piece_dict))
    if not verbose_off:
        print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, \
        help='Path to folder containing taxon-labelled subfolders with corresponding greyscale images for input.')
    parser.add_argument('output_path', type=str, \
        help='Path to folder that images will be output to. Subfolders are created automatically.')
    
    parser.add_argument('-v', '--version', action='version', version=version)
    
    parser.add_argument('--best_must_pass', default=best_must_pass, action='store_true', \
        help='By default, the "best" image is always output even if it doesn\'t pass std_threshold or min_dim_scalar, but if you include this \
                tag it will ignore everything that doesn\'t pass either, potentially not outputting anything for some input images.')
    parser.add_argument('--best_only', default=best_only, action='store_true', \
        help='Include this tag to only output the "best" image found via the panning algorithm, as opposed to multiple.')
    parser.add_argument('--c_size', default=c_size, type=int, \
        help='Dimensions of boxes taken from the image corners used to calculate the background median and standard deviation values \
                for stochastic padding. Default value is '+str(c_size)+'.')
    parser.add_argument('--desired_size', default=desired_size, type=int, \
        help='The desired width and height of the output image. Note that the ifcb_classifier uses '+str(desired_size)+'x'+str(desired_size)+' images, \
                which is the default here.')
    parser.add_argument('--gaussian_sigma', default=gaussian_sigma, type=int, \
        help='The sigma value for optional Gaussian smoothing. Default value is 0 (off).')
    parser.add_argument('--hop_scalar', default=hop_scalar, type=float, \
        help='The ceiling of this number multiplied by desired_size is the number of pixels that will be hopped to crop neighbouring images \
                when outputting multiple. Default is '+str(hop_scalar)+'.')
    parser.add_argument('--invert', default=invert, action='store_true', \
        help='Include this tag to invert the image.')
    parser.add_argument('--min_dim_scalar', default=min_dim_scalar, type=float, \
        help='This number multiplied by desired size is the smallest dimension an image can have in order to be output. \
                Default is '+str(min_dim_scalar)+'.')
    parser.add_argument('--p_samples', default=p_samples, type=int, \
        help='The number of random coordinates generated to find the optimal panning placement. Default value is '+str(p_samples)+'.')
    parser.add_argument('--rand_scalar', default=rand_scalar, type=float, \
        help='The additional scalar the pixel values are multiplied by for stochastic padded. Default value is '+str(rand_scalar)+'.')
    parser.add_argument('--resize_scalar', default=resize_scalar, type=float, \
        help='Images will be initially re-sized by multiplying their dimensions by the input number - more or less zooming in or out. \
                Default value is 1 (off).')
    parser.add_argument('--std_threshold', default=std_threshold, type=float, \
        help='The threshold that the standard deviation of the normalized pixel values of an image that must be crossed in order for said image \
                to be output. Default is '+str(std_threshold)+'.')
    parser.add_argument('--use_var', default=use_var, action='store_true', \
        help='Include this tag to multiply the random value for each pixel by the variance of the \'corner boxes\' instead of the standard \
                deviation for stochastic padding.')
    parser.add_argument('--verbose_off', default=verbose_off, action='store_true', \
        help='Include this tag to disable print statements.')
    
    args = parser.parse_args()
    main(args)