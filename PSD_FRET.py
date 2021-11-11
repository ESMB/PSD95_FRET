#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:27:53 2021

@author: Mathew
"""

from skimage.io import imread
import os
import pandas as pd
from picasso import render
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from skimage import filters,measure
from skimage.filters import threshold_local
import math

donor_filename="Donor.tif"
# Direct excitation:
acceptor_filename="Acceptor.tif"
# Excitation by FRET
FRET_filename="FRET.tif"

filename_contains="647_0"

# Folders to analyse:

pathList=[]

pathList.append("/Users/Mathew/Documents/Current analysis/PSD_FRET/CA1/")
pathList.append("/Users/Mathew/Documents/Current analysis/PSD_FRET/SSC_L23/")
pathList.append("/Users/Mathew/Documents/Current analysis/PSD_FRET/Thal/")
pathList.append("/Users/Mathew/Documents/Current analysis/PSD_FRET/Striatum/")
pathList.append("/Users/Mathew/Documents/Current analysis/PSD_FRET/Midbrain/")
pathList.append("/Users/Mathew/Documents/Current analysis/PSD_FRET/SSCL5/")


def load_image(toload):
    
    image=imread(toload)
    
    return image

def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)    
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# This is to look at coincidence purely in terms of pixels

def coincidence_analysis_pixels(binary_image1,binary_image2):
    pixel_overlap_image=binary_image1&binary_image2         
    pixel_overlap_count=pixel_overlap_image.sum()
    pixel_fraction=pixel_overlap_image.sum()/binary_image1.sum()
    
    return pixel_overlap_image,pixel_overlap_count,pixel_fraction


def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return labelled_image


for path in pathList:
    
    # Load the images
    Donor_image=load_image(path+donor_filename)
    Acceptor_image=load_image(path+acceptor_filename)
    FRET_image=load_image(path+FRET_filename)
    
    # Binarise the donor and acceptor images
    Donor_binary_threshdold,Donor_binary_image=threshold_image_otsu(Donor_image)
    Acceptor_binary_threshdold,Acceptor_binary_image=threshold_image_otsu(Acceptor_image)
    
    # Save the binary images
    im = Image.fromarray(Donor_binary_image)
    im.save(path+'Donor_binary.tif')
    im = Image.fromarray(Acceptor_binary_image)
    im.save(path+'Acceptor_binary.tif')

    # Now look for coincident pixels
    pixel_overlap_image,pixel_overlap_count,pixel_fraction=coincidence_analysis_pixels(Donor_binary_image,Acceptor_binary_image)
    
    # Save the binary coincident_image
    im = Image.fromarray(pixel_overlap_image)
    im.save(path+'Coinc_binary.tif')

    # Calculate proximity ratio for all of images
    Proximity_image=FRET_image/(FRET_image+Donor_image)
    Proximity_image_coincident_only=Proximity_image*pixel_overlap_image
    
    # Save the FRET images
    im = Image.fromarray(Proximity_image)
    im.save(path+'Proximity.tif')
    im = Image.fromarray(Proximity_image_coincident_only)
    im.save(path+'Proximity_thresholded.tif')
    
    # Perform analysis of the FRET efficiencies
    labelled_coincident_image=label_image(pixel_overlap_image)
    FRET_analysis=analyse_labelled_image(labelled_coincident_image,Proximity_image)
    # Save the image and analysis
    FRET_analysis.to_csv(path + 'FRET_metrics.csv', sep = '\t') 
    im = Image.fromarray(labelled_coincident_image)
    im.save(path+'Labelled.tif')
    

    # Generate histogram of FRET efficiences
    FRET=FRET_analysis['mean_intensity']
    plt.hist(FRET, bins = 50,range=[0,1], rwidth=0.9,color='#ff0000')
    plt.xlabel('Proximity ratio',size=20)
    plt.ylabel('Number of PSDs',size=20)
    plt.title('Promximity ratio')
    plt.savefig(path+'FRET Histogram.pdf')
    plt.show()
    
    
    # Perform analysis of the intensity ratios
    lnz_image=np.log(Acceptor_image/Donor_image)
    lnz_analysis=analyse_labelled_image(labelled_coincident_image,lnz_image)
    # Save the image and analysis
    lnz_analysis.to_csv(path + 'Ratio_metrics.csv', sep = '\t') 
  

    # Generate histogram of lnz
    lnz=lnz_analysis['mean_intensity']
    plt.hist(lnz, bins = 50,range=[-3,0], rwidth=0.9,color='#ff0000')
    plt.xlabel('ln(acceptor/donor)',size=20)
    plt.ylabel('Number of PSDs',size=20)
    plt.title('Z=ln(acceptor/donor)')
    plt.savefig(path+'Ratio Histogram.pdf')
    plt.show()
    
     # Perform analysis of the donor intensity
   
    don_analysis=analyse_labelled_image(labelled_coincident_image,Donor_image)
    # Save the image and analysis
    don_analysis.to_csv(path + 'Donor_metrics.csv', sep = '\t') 
  

    # Generate histogram of lnz
    don=don_analysis['mean_intensity']
    plt.hist(don, bins = 50,range=[0,8000], rwidth=0.9,color='#ff0000')
    plt.xlabel('Donor intensity',size=20)
    plt.ylabel('Number of PSDs',size=20)
    plt.title('Donor Intensity')
    plt.savefig(path+'Donor Histogram.pdf')
    plt.show()
    
    acc_analysis=analyse_labelled_image(labelled_coincident_image,Acceptor_image)
    # Save the image and analysis
    acc_analysis.to_csv(path + 'Acceptor_metrics.csv', sep = '\t') 
  

    # Generate histogram of lnz
    acc=acc_analysis['mean_intensity']
    plt.hist(acc, bins = 50,range=[0,8000], rwidth=0.9,color='#ff0000')
    plt.xlabel('Acceptor intensity',size=20)
    plt.ylabel('Number of PSDs',size=20)
    plt.title('Acceptor Intensity')
    plt.savefig(path+'Acceptor Histogram.pdf')
    plt.show()
    
    FRETintensity_analysis=analyse_labelled_image(labelled_coincident_image,FRET_image)
    # Save the image and analysis
    FRETintensity_analysis.to_csv(path + 'FRETintensity_metrics.csv', sep = '\t') 
  

    # Generate histogram of lnz
    FRETintensity=FRETintensity_analysis['mean_intensity']
    plt.hist(FRETintensity, bins = 50,range=[0,8000], rwidth=0.9,color='#ff0000')
    plt.xlabel('FRET intensity',size=20)
    plt.ylabel('Number of PSDs',size=20)
    plt.title('FRET Intensity')
    plt.savefig(path+'FRET intensity Histogram.pdf')
    plt.show()
    
    
    
    