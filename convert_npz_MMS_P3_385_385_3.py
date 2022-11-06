import os
import numpy as np
from numpy import asarray,savez_compressed
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import argparse

#load all images in a directory into memory
def load_images(imgpath,maskpath,labelpath,n_crops, size=(128,128)):
    src_list, mask_list, label_list = list(), list(), list()

    input_file_names = os.listdir(imgpath)

    for input_file_name in input_file_names:
        # load and resize the image
        filename = input_file_name
        mask_name = input_file_name.replace('_P3_', '_P3_mask_')
        label_name = input_file_name.replace('_P3_', '_P3_label_')
        
        img = load_img(imgpath + filename, target_size=size)
        fundus_img = img_to_array(img)

        mask = load_img(maskpath + mask_name, target_size=size,color_mode="grayscale")
        mask_img = img_to_array(mask)
        
        label = load_img(labelpath + label_name, target_size=size,color_mode="grayscale")
        label_img = img_to_array(label)
        
        # split into satellite and map
        src_list.append(fundus_img)
        mask_list.append(mask_img)
        label_list.append(label_img)
    return [asarray(src_list), asarray(mask_list), asarray(label_list)]
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=(128,128))
    parser.add_argument('--n_crops', type=int, default=210)
    parser.add_argument('--outfile_name', type=str, default='MMS_P3_385_385')
    args = parser.parse_args()

    # dataset path
    imgpath = 'MMS_P3_385_385_3_crop/Images/'
    maskpath = 'MMS_P3_385_385_3_crop/Masks/'
    labelpath = 'MMS_P3_385_385_3_crop/labels/'
    # load dataset
    [src_images, mask_images, label_images] = load_images(imgpath,maskpath,labelpath,args.n_crops,args.input_dim)
    print('Loaded: ', src_images.shape, mask_images.shape, label_images.shape)
    # save as compressed numpy array
    filename = args.outfile_name+'.npz'
    savez_compressed(filename, src_images, mask_images, label_images)
    print('Saved dataset: ', filename)
