# -*- coding: utf-8 -*-
#Import Dependencies
import numpy as np
import dicom
import os
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans


# Configuration Variables
# TODO : set the configuration parameters in a conf file
data_path = "C:\\Users\\Documents\\Kaggle\\Patients_new\\"
output_path = working_path = "C:\\Users\\Documents\\Kaggle\\Output\\"

#Set shape of the resample volume in order to have consistant volume size (could be modified for accuracy)
#this is due to the fact that computed resize factor lead to different output boxes size.
#Discarded : 
NEW_SHAPE = np.array([300,310,310])

#Array list of every patients in the folder.
#TODO: need to buffer if the patient list is too long 
PATIENTS = os.listdir(data_path)


# Function to load a scan images
# IN  : path (from configuration file)
# OUT : All the DICOM objects from the patient in path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

#Function
# IN  : All the DICOM objects from the patient in path
# OUT : Hounsfield unit array for every CT Scan
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # Should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image,dtype=np.int16)

#Function
# IN  : Hounsfield unit array for every CT Scan
# OUT : Resampled inputs according to new spacing paramter, new spacing
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    
    # Computing the resize factor 
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    # Dimensions of the cube are larger than the input cube due to zooming 
    return image, new_spacing

#Standardize the pixel values
def make_lungmask(img):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    #label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    segmented_ratio = float(sum(mask[mask>0])/(row_size*col_size))
    print("Ratio of unmasked_pixel over full frame : " +str(segmented_ratio))
    
    return mask*img



for patient in PATIENTS:
    
    patients_path = data_path + str(patient)
    patient_output_path = output_path + str(patient) + str("\\")
  
    try : 
        os.mkdir(patient_output_path)
    # Catch both errors in case the script is run on unix or windows     
    except (OSError,FileExistsError):
        #TODO : Deal with OSErrors other than file exists  
        pass
     
    # Gets all the scans from a patient for processing 
    scans = load_scan(patients_path)
    # Transforms scans into houndfield units 2D pixel_array 
    scans_hu = get_pixels_hu(scans)
    # Resampling of the scans (modifies the shape of the input images)
    scans_rs,spacing = resample(scans_hu,scans,[1,1,1])
    #Saves the resampled arrays wheter further segmentation is required 
    print('Resampled Outputs path : ' + str(patient_output_path))
    print('Shape of the outputs   : ' + str(scans_rs.shape))
    np.save(patient_output_path + str(patient) + ".npy",scans_rs)  
    #Segmentation of the resampled scans using mask technique
    #import matplotlib.pyplot as plt
    segmented_scan = np.stack([make_lungmask(i) for i in scans_rs])
    
    #plt.imshow(segmented_scan, cmap='gray', interpolation='nearest')
    np.save(patient_output_path + str(patient) + "_segmented.npy",segmented_scan)  
    
    
    
