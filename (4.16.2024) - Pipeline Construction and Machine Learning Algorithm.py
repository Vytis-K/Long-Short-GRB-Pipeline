#!/usr/bin/env python
# coding: utf-8

# In[7]:


from gbm.data import Trigdat

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from gbm.plot import Lightcurve

from gbm import test_data_dir
from gbm.data import TTE
from gbm.plot import Spectrum
import matplotlib.pyplot as plt
from gbm.data.primitives import TimeBins, EnergyBins
from gbm.data.primitives import TimeEnergyBins

from gbm.binning.unbinned import bin_by_time
import numpy as np

from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import pywt
from scipy.ndimage import binary_erosion, binary_dilation

from scipy.optimize import curve_fit

from gbm.finder import TriggerCatalog, TriggerFtp
import os


# Broken portion of pipeline.

# In[3]:


def polynomial(x, *coeffs):
    return np.polyval(coeffs, x)

def subtract_background_iteratively(bin_times, bin_counts, degree=2, iterations=2):
    corrected_counts = bin_counts.copy()
    for iteration in range(iterations):
        for i in range(corrected_counts.shape[0]):
            coeffs, _ = curve_fit(polynomial, bin_times, corrected_counts[i, :], p0=[1] * (degree + 1))
            background = polynomial(bin_times, *coeffs)
            corrected_counts[i, :] -= background
            corrected_counts[i, :] = np.clip(corrected_counts[i, :], 0, None)

    noise_level = np.std(corrected_counts, axis=1)
    snr_threshold = 2
    for i in range(corrected_counts.shape[0]):
        signal = corrected_counts[i, :]
        noise = noise_level[i]
        corrected_counts[i, :] = np.where(signal > snr_threshold * noise, signal, 0)

    return corrected_counts

def process_tte_data(detector_name, bin_number, time_range, bin_width, smearing_factor=0.1):
    try:
        tte = TTE.open(f'glg_tte_{detector_name}_{bin_number}_v00.fit')
        time_sliced_tte = tte.slice_time(time_range)
        eventlist = time_sliced_tte.data

        #bin_width = 1.024
        bins = eventlist.bin(bin_by_time, bin_width)
        bin_times = bins.time_centroids
        bin_counts = bins.counts
        energy = bins.energy_centroids

        # Smear the bin counts
        smeared_counts = np.copy(bin_counts)
        for i in range(1, len(bin_counts) - 1):
            smeared_counts[i] = (smearing_factor * bin_counts[i - 1] +
                                 (1 - 2 * smearing_factor) * bin_counts[i] +
                                 smearing_factor * bin_counts[i + 1])

        return bin_times, smeared_counts, energy
    except AttributeError as e:
        print(f"Error processing TTE data for {detector_name}: {e}")
        return None, None, None  # Return None values to indicate failure
    
time_ranges = [(-15, 15), (-50, 50), (-150, 150)]  

# Calculate dynamic bin size
base_time_range = 300  # Reference time range for a bin width of 1.024
base_bin_width = 1.024

# Step 1: Choose GRBs of Interest
trigcat = TriggerCatalog()
sliced_trigcat = trigcat.slice('trigger_type')

# Step 2: Initialize Trigger Finder
for trigger_info in sliced_trigcat.get_table(columns=('trigger_name', 'trigger_time')).tolist()[:100]:
    trigger_name, trigger_time = trigger_info
    print(f"Downloading TTE files for GRB {trigger_name} ({trigger_time})")
    
    # Remove "bn" from trigger_name
    modified_trigger_name = trigger_name.replace("bn", "")
    trig_finder = TriggerFtp(modified_trigger_name)

    # Download trigdat file to the current directory
    try:
        trig_finder.get_trigdat('./')
    except AttributeError as e:
        print(f"AttributeError occurred while downloading trigdat file for GRB {trigger_name}: {e}")
        continue
      
    # Try opening the file with v01, if not found, try v00 and v02
    version_suffixes = ['v01', 'v00', 'v02']
    for suffix in version_suffixes:
        filename = f'glg_trigdat_all_{trigger_name}_{suffix}.fit'
        try:
            trigdat = Trigdat.open(filename)
            trig_dets = trigdat.triggered_detectors
            file_found = True
            break
        except OSError as e:
            if "does not exist" in str(e):
                print(f"File {filename} does not exist. Trying next version.")
            else:
                raise

    if not file_found:
        print("None of the files with the specified versions were found.")
    
    # Download TTE files to the current directory
    trig_finder.get_tte('./')

    print(f"Download complete for GRB {trigger_name}\n")

    for time_range in time_ranges:
        
        duration = time_range[1] - time_range[0]
        bin_width = (duration / base_time_range) * base_bin_width
        
        time_range_dir = f'/Users/vytis/Desktop/GRB Images/TimeRange_{time_range[0]}_{time_range[1]}'
        os.makedirs(time_range_dir, exist_ok=True)
        
        # Process the downloaded TTE data
        bin_number = modified_trigger_name
        if not bin_number.startswith('bn'):
            bin_number = 'bn' + bin_number
        length = "Long"
        grb_name = bin_number.replace("bn", "")

        # NaI Detectors
        na_detectors = trig_dets
        na_counts_list = []
        for detector in na_detectors:
            bin_times, bin_counts, _ = process_tte_data(detector, bin_number, time_range, bin_width)

            # Remove the bottom 90% of counts for each column
            percentile_90 = np.percentile(bin_counts, 90, axis=0)
            for i in range(bin_counts.shape[1]):
                bin_counts[:, i] = np.where(bin_counts[:, i] > percentile_90[i], bin_counts[:, i], 0)

            #bin_counts_nai_corrected = subtract_background(bin_times_nai, bin_counts_nai.T).T
            bin_counts_polynomial_filtering = subtract_background_iteratively(bin_times, bin_counts.T).T

            # Set the last two rows to 0
            bin_counts_polynomial_filtering[:, -3:] = 0
            
            # Normalize the filtered counts
            bin_counts_normalized = bin_counts_polynomial_filtering / np.max(bin_counts_polynomial_filtering)

            # Apply median filter
            filtered_counts = median_filter(bin_counts_normalized, size=(2, 2))

            na_counts_list.append(filtered_counts)

        try:
            # Combine processed counts from all NaI detectors
            bin_counts_nai = np.sum(na_counts_list, axis=0)
            filtered_counts_log_nai = np.log1p(bin_counts_nai)
        except ValueError as e:
            print(f"ValueError occurred while combining detector counts: {e}")
            # Handle the error case by skipping the current loop iteration
            continue

        bin_times_list, _, energy_list = zip(*[process_tte_data(detector, bin_number, time_range) for detector in na_detectors])

        # Combine counts from all NaI detectors
        bin_times_nai = bin_times_list[0]
        energy_nai = energy_list[0]

        # Bismuth Detectors
        bgo_detectors = ['b0', 'b1']
        bgo_counts_list = []
        for detector in bgo_detectors:
            bin_times, bin_counts, _ = process_tte_data(detector, bin_number, time_range)

            try:
                percentile_90 = np.percentile(bin_counts, 90, axis=0)
                for i in range(bin_counts.shape[1]):
                    bin_counts[:, i] = np.where(bin_counts[:, i] > percentile_90[i], bin_counts[:, i], 0)
            except np.AxisError as e:
                print(f"AxisError occurred while processing percentile for detector {detector}: {e}")
                continue

            # Normalize the filtered counts
            bin_counts_normalized = bin_counts / np.max(bin_counts)

            # Apply median filter
            filtered_counts = median_filter(bin_counts_normalized, size=(2, 2))

            bgo_counts_list.append(filtered_counts)

        # Combine processed counts from all NaI detectors
        bin_counts_bgo = np.sum(bgo_counts_list, axis=0)
        filtered_counts_log_bgo = np.log1p(bin_counts_bgo)

        bin_times_bgo, bin_counts_bgo, energy_bgo = zip(*[process_tte_data(detector, bin_number, time_range) for detector in bgo_detectors])

        # Combine counts from all BGO detectors
        bin_times_bgo_combined = bin_times_bgo[0]
        energy_bgo_combined = energy_bgo[0]

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        # Plot for NaI Detectors
        im1 = ax1.imshow(filtered_counts_log_bgo.T, aspect='auto', extent=[bin_times_bgo_combined.min(), bin_times_bgo_combined.max(), energy_bgo_combined.min(), energy_bgo_combined.max()], cmap='viridis', origin='lower')
        ax1.set_ylabel('Energy (keV)')
        ax1.set_title(f'GRB {grb_name} ({length}) Count Heat Map for Time Range {time_range[0]} to {time_range[1]}')
        fig.colorbar(im1, ax=ax1, label='Log(Counts + 1)')
        
        # Plot for BGO Detectors
        im2 = ax2.imshow(filtered_counts_log_nai.T, aspect='auto', extent=[bin_times_nai.min(), bin_times_nai.max(), energy_nai.min(), energy_nai.max()], cmap='viridis', origin='lower')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy (keV)')
        ax2.set_title('')
        fig.colorbar(im2, ax=ax2, label='Log(Counts + 1)')

        plt.subplots_adjust(hspace=0.01)
        plt.savefig(os.path.join(time_range_dir, f'GRB_{grb_name}_TimeRange_{time_range[0]}_{time_range[1]}.png'))
        plt.close()
        print(f'Image for time range {time_range[0]} to {time_range[1]} successfully created.')


# Things changed: bin width dynamically adjusts relative to a benchmark "medium width" (maybe change it depending on whether it's a short or long event?) with a base 1.024s time interval.
# Updated error catching and changed trigger indexing.

# In[15]:


def polynomial(x, *coeffs):
    try:
        return np.polyval(coeffs, x)
    except Exception as e:
        print(f"Error in polynomial calculation: {e}")
        return 0

def subtract_background_iteratively(bin_times, bin_counts, degree=2, iterations=2):
    corrected_counts = bin_counts.copy()
    for iteration in range(iterations):
        for i in range(corrected_counts.shape[0]):
            try:
                coeffs, _ = curve_fit(polynomial, bin_times, corrected_counts[i, :], p0=[1] * (degree + 1))
                background = polynomial(bin_times, *coeffs)
                corrected_counts[i, :] -= background
                corrected_counts[i, :] = np.clip(corrected_counts[i, :], 0, None)
            except Exception as e:
                print(f"Error during background subtraction on iteration {iteration}, row {i}: {e}")
                continue
    try:
        noise_level = np.std(corrected_counts, axis=1)
        snr_threshold = 2
        for i in range(corrected_counts.shape[0]):
            signal = corrected_counts[i, :]
            noise = noise_level[i]
            corrected_counts[i, :] = np.where(signal > snr_threshold * noise, signal, 0)
    except Exception as e:
        print(f"Error calculating SNR and updating counts: {e}")

    return corrected_counts

def process_tte_data(detector_name, bin_number, time_range, bin_width, smearing_factor=0.1):
    try:
        tte = TTE.open(f'glg_tte_{detector_name}_{bin_number}_v00.fit')
        time_sliced_tte = tte.slice_time(time_range)
        eventlist = time_sliced_tte.data

        #bin_width = 1.024
        bins = eventlist.bin(bin_by_time, bin_width)
        bin_times = bins.time_centroids
        bin_counts = bins.counts
        energy = bins.energy_centroids

        # Smear the bin counts
        smeared_counts = np.copy(bin_counts)
        for i in range(1, len(bin_counts) - 1):
            smeared_counts[i] = (smearing_factor * bin_counts[i - 1] +
                                 (1 - 2 * smearing_factor) * bin_counts[i] +
                                 smearing_factor * bin_counts[i + 1])

        return bin_times, smeared_counts, energy
    except OSError as e:
        print(f"Error processing TTE data for {detector_name}: {e}")
        return None, None, None  # Return None values to indicate failure
    except AttributeError as e:
        print(f"Error processing TTE data for {detector_name}: {e}")
        return None, None, None  # Return None values to indicate failure
    except Exception as e:
        print(f"Unhandled exception: {e}")
        return None, None, None
    
time_ranges = [(-15, 15), (-50, 50), (-150, 150)]  

# Calculate dynamic bin size
base_time_range = 300  # Reference time range for a bin width of 1.024
base_bin_width = 1.024

# Step 1: Choose GRBs of Interest
trigcat = TriggerCatalog()
sliced_trigcat = trigcat.slice('trigger_type')

# Step 2: Initialize Trigger Finder
for trigger_info in sliced_trigcat.get_table(columns=('trigger_name', 'trigger_time')).tolist()[:100]:
    trigger_name, trigger_time = trigger_info
    print(f"Downloading TTE files for GRB {trigger_name} ({trigger_time})")
    
    try:
        # Remove "bn" from trigger_name
        modified_trigger_name = trigger_name.replace("bn", "")
        trig_finder = TriggerFtp(modified_trigger_name)
    
        try:
            # Try to list files, which could throw an SSLEOFError
            trig_finder._file_list = trig_finder.ls(modified_trigger_name)
            trig_finder._ftp.cwd(trig_finder._construct_path(modified_trigger_name))
            
        except SSLEOFError as ssl_error:
            print(f"SSL EOF Error: {ssl_error}")
            continue
            
        except FileExistsError as fe_error:
            print(f"File not found on FTP server: {fe_error}")
            continue
            
    except ValueError as ve:
        print(f"ValueError: {ve}")
        continue
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        continue

    try:
        trig_finder.get_trigdat('./')
    except AttributeError as e:
        print(f"AttributeError occurred while downloading trigdat file for GRB {trigger_name}: {e}")
        continue
      
    # Try opening the file with v01, if not found, try v00 and v02
    version_suffixes = ['v01', 'v00', 'v02']
    for suffix in version_suffixes:
        filename = f'glg_trigdat_all_{trigger_name}_{suffix}.fit'
        try:
            trigdat = Trigdat.open(filename)
            trig_dets = trigdat.triggered_detectors
            file_found = True
            break
        except IndexError as ie:
            print(f"Index error: No data available in the file {filename}. {ie}")
            continue
        except OSError as e:
            if "does not exist" in str(e):
                print(f"File {filename} does not exist. Trying next version.")
            else:
                raise

    if not file_found:
        print("None of the files with the specified versions were found.")
    
    # Download TTE files to the current directory
    trig_finder.get_tte('./')

    print(f"Download complete for GRB {trigger_name}\n")

    for time_range in time_ranges:
        
        duration = time_range[1] - time_range[0]
        bin_width = (duration / base_time_range) * base_bin_width
        
        time_range_dir = f'/Users/vytis/Desktop/GRB Images/TimeRange_{time_range[0]}_{time_range[1]}'
        os.makedirs(time_range_dir, exist_ok=True)
        
        # Process the downloaded TTE data
        bin_number = modified_trigger_name
        if not bin_number.startswith('bn'):
            bin_number = 'bn' + bin_number
        length = "Long"
        grb_name = bin_number.replace("bn", "")

        # NaI Detectors
        na_detectors = trig_dets
        na_counts_list = []
        for detector in na_detectors:
            bin_times, bin_counts, _ = process_tte_data(detector, bin_number, time_range, bin_width)

            # Remove the bottom 90% of counts for each column
            try:
                percentile_90 = np.percentile(bin_counts, 90, axis=0)
                for i in range(bin_counts.shape[1]):
                    bin_counts[:, i] = np.where(bin_counts[:, i] > percentile_90[i], bin_counts[:, i], 0)
            except np.AxisError as e:
                print(f"AxisError occurred while processing percentile for detector {detector}: {e}")
                continue

            #bin_counts_nai_corrected = subtract_background(bin_times_nai, bin_counts_nai.T).T
            bin_counts_polynomial_filtering = subtract_background_iteratively(bin_times, bin_counts.T).T

            # Set the last two rows to 0
            bin_counts_polynomial_filtering[:, -3:] = 0
            
            # Normalize the filtered counts
            bin_counts_normalized = bin_counts_polynomial_filtering / np.max(bin_counts_polynomial_filtering)

            # Apply median filter
            filtered_counts = median_filter(bin_counts_normalized, size=(2, 2))

            na_counts_list.append(filtered_counts)

        try:
            # Combine processed counts from all NaI detectors
            bin_counts_nai = np.sum(na_counts_list, axis=0)
            filtered_counts_log_nai = np.log1p(bin_counts_nai)
        except ValueError as e:
            print(f"ValueError occurred while combining detector counts: {e}")
            # Handle the error case by skipping the current loop iteration
            continue

        bin_times_list, _, energy_list = zip(*[process_tte_data(detector, bin_number, time_range, bin_width) for detector in na_detectors])

        # Combine counts from all NaI detectors
        bin_times_nai = bin_times_list[0]
        energy_nai = energy_list[0]

        # Bismuth Detectors
        bgo_detectors = ['b0', 'b1']
        bgo_counts_list = []
        for detector in bgo_detectors:
            bin_times, bin_counts, _ = process_tte_data(detector, bin_number, time_range, bin_width)

            try:
                percentile_90 = np.percentile(bin_counts, 90, axis=0)
                for i in range(bin_counts.shape[1]):
                    bin_counts[:, i] = np.where(bin_counts[:, i] > percentile_90[i], bin_counts[:, i], 0)
            except np.AxisError as e:
                print(f"AxisError occurred while processing percentile for detector {detector}: {e}")
                continue

            # Normalize the filtered counts
            bin_counts_normalized = bin_counts / np.max(bin_counts)

            # Apply median filter
            filtered_counts = median_filter(bin_counts_normalized, size=(2, 2))

            bgo_counts_list.append(filtered_counts)

        try:
            # First, check if all arrays in bgo_counts_list have the same shape
            if not all(x.shape == bgo_counts_list[0].shape for x in bgo_counts_list):
                raise ValueError("All arrays in bgo_counts_list must have the same shape to sum.")    
        
            # Combine processed counts from all NaI detectors
            bin_counts_bgo = np.sum(bgo_counts_list, axis=0)
            filtered_counts_log_bgo = np.log1p(bin_counts_bgo)

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred while summing BGO counts: {e}")

        bin_times_bgo, bin_counts_bgo, energy_bgo = zip(*[process_tte_data(detector, bin_number, time_range, bin_width) for detector in bgo_detectors])

        # Combine counts from all BGO detectors
        bin_times_bgo_combined = bin_times_bgo[0]
        energy_bgo_combined = energy_bgo[0]

        try:
            # Check if bin_times_bgo_combined or energy_bgo_combined are None before plotting
            if bin_times_bgo_combined is None or energy_bgo_combined is None:
                raise ValueError("Time or energy data is missing for BGO detectors.")
            if bin_times_nai is None or energy_nai is None:
                raise ValueError("Time or energy data is missing for NaI detectors.")
        
            # Create the figure
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

            # Plot for NaI Detectors
            im1 = ax1.imshow(filtered_counts_log_bgo.T, aspect='auto', extent=[bin_times_bgo_combined.min(), bin_times_bgo_combined.max(), energy_bgo_combined.min(), energy_bgo_combined.max()], cmap='viridis', origin='lower')
            ax1.set_ylabel('Energy (keV)')
            ax1.set_title(f'GRB {grb_name} ({length}) Count Heat Map for Time Range {time_range[0]} to {time_range[1]}')
            fig.colorbar(im1, ax=ax1, label='Log(Counts + 1)')

            # Plot for BGO Detectors
            im2 = ax2.imshow(filtered_counts_log_nai.T, aspect='auto', extent=[bin_times_nai.min(), bin_times_nai.max(), energy_nai.min(), energy_nai.max()], cmap='viridis', origin='lower')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Energy (keV)')
            ax2.set_title('')
            fig.colorbar(im2, ax=ax2, label='Log(Counts + 1)')

            plt.subplots_adjust(hspace=0.01)
            plt.savefig(os.path.join(time_range_dir, f'GRB_{grb_name}_TimeRange_{time_range[0]}_{time_range[1]}.png'))
            plt.close()
            print(f'Image for time range {time_range[0]} to {time_range[1]} successfully created.')
            
        except ValueError as ve:
            print(f"Value error: {ve}")

        except AttributeError as ae:
            print(f"Attribute error: Data might be incomplete or improperly formatted. {ae}")

        except Exception as e:
            print(f"An unexpected error occurred while trying to plot: {e}")


# In[16]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import backend as K


# In[17]:


K.set_image_data_format('channels_last')

# Load the ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet', include_top=False)

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Convert similarity score to percentage
def similarity_to_percentage(score):
    return (score + 1) / 2 * 100

# Function to process the matching for given time ranges
def process_matching_for_time_range(time_range):
    training_dir = f'/Users/vytis/Desktop/GRB Images/Training Data/TimeRange_{time_range[0]}_{time_range[1]}'
    alternative_dir = f'/Users/vytis/Desktop/GRB Images/TimeRange_{time_range[0]}_{time_range[1]}'

    # Extract features from training images
    training_features = []
    training_image_paths = [os.path.join(training_dir, filename) for filename in os.listdir(training_dir)]
    for img_path in training_image_paths:
        training_features.append(extract_features(img_path, model))
    training_features = np.array(training_features)

    # Extract features from alternative images and compare with training features
    alternative_image_paths = [os.path.join(alternative_dir, filename) for filename in os.listdir(alternative_dir)]
    
    for alt_img_path in alternative_image_paths:
        alt_features = extract_features(alt_img_path, model)
        similarities = cosine_similarity([alt_features], training_features)
        most_similar_index = np.argmax(similarities)
        most_similar_image_path = training_image_paths[most_similar_index]
        percent_match = similarity_to_percentage(similarities[0][most_similar_index])
        print(f"Time Range {time_range}: The image {os.path.basename(alt_img_path)} matches {os.path.basename(most_similar_image_path)} by {percent_match:.2f}%.")

# List of time ranges to process
time_ranges = [(-15, 15), (-50, 50), (-150, 150)]

# Process each time range
for time_range in time_ranges:
    process_matching_for_time_range(time_range)


# In[9]:


import scipy.stats as stats

prob_below_100 = stats.norm.cdf(-0.4)
prob_below_100


# In[ ]:




