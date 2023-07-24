import os
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import time
from brainiak.searchlight.searchlight import Searchlight
from scipy.stats import spearmanr

"""
# Beta values
!wget -O betas_1.nii.gz https://www.dropbox.com/s/5f17f5fk8merind/sub-01_space-T1w_desc-train-fracridge_data.nii.gz?dl=0
!wget -O betas_2.nii.gz https://www.dropbox.com/s/763e41z0no2nhxq/sub-02_space-T1w_desc-train-fracridge_data.nii.gz?dl=0
!wget -O betas_3.nii.gz https://www.dropbox.com/s/igdj59u6opny8as/sub-03_space-T1w_desc-train-fracridge_data.nii.gz?dl=0
!wget -O betas_4.nii.gz https://www.dropbox.com/s/pb6bx4sj3j3266c/sub-04_space-T1w_desc-train-fracridge_data.nii.gz?dl=0

# Anatomical brains
!wget -O anat_1.nii.gz https://www.dropbox.com/s/1otl3clmpz0dgm5/sub-01_desc-preproc_T1w.nii.gz?dl=0
!wget -O anat_2.nii.gz https://www.dropbox.com/s/h0r2lufwqxi2t1e/sub-02_desc-preproc_T1w.nii.gz?dl=0
!wget -O anat_3.nii.gz https://www.dropbox.com/s/ocfl19d5e0xj2xi/sub-03_desc-preproc_T1w.nii.gz?dl=0
!wget -O anat_4.nii.gz https://www.dropbox.com/s/kpjm47p83t6mhn3/sub-04_desc-preproc_T1w.nii.gz?dl=0
"""

print("Files Loaded!")

# Load the AlexNet model layers for rsa
model_layer1 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.0.npz')['arr_0']
model_layer2 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.3.npz')['arr_0']
model_layer3 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.6.npz')['arr_0']
model_layer4 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.8.npz')['arr_0']
model_layer5 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.10.npz')['arr_0']

layer1_lower = model_layer1[np.tril_indices_from(model_layer1, k=-1)]
layer2_lower = model_layer2[np.tril_indices_from(model_layer2, k=-1)]
layer3_lower = model_layer3[np.tril_indices_from(model_layer3, k=-1)]
layer4_lower = model_layer4[np.tril_indices_from(model_layer4, k=-1)]
layer5_lower = model_layer5[np.tril_indices_from(model_layer5, k=-1)]

layers = [layer1_lower, layer2_lower, layer3_lower, layer4_lower, layer5_lower]

# Set up the kernel function for RSA
def calc_rsa(data, sl_mask, myrad, bcvar):
    # Pull out the data
    data4D = data[0]
    bolddata_sl = data4D.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data[0].shape[3])

    # Create the searchlight rdm
    df_beta = pd.DataFrame(bolddata_sl)
    df_pearson = 1 - df_beta.corr(method='pearson')
    sub_rdm = df_pearson.to_numpy()
    lower = sub_rdm[np.tril_indices_from(sub_rdm, k=-1)]

    # perform spearman correlation
    corr = spearmanr(lower, layer1_lower)[0]
    return corr


# Searchlight
for sub in range(1, 5):
    layer = '1'
    bold_vol = np.load("/content/rsatoolbox/isiktoolbox/data/RoiSubjectBetas/whole_brain_array_" + str(sub) + ".npz")[
        'arr_0']
    whole_brain_mask = np.zeros(bold_vol.shape[0:3])
    affine_mat = nib.load("/content/rsatoolbox/isiktoolbox/data/RoiSubjectBetas/betas_" + str(sub) + ".nii.gz").affine

    # Make a whole brain mask
    whole_mask = np.ones(whole_brain_mask.shape)

    # Preset the variables
    data = bold_vol
    mask = whole_mask
    bcvar = None
    sl_rad = 1
    max_blk_edge = 5
    pool_size = 1

    # Start the clock to time searchlight
    begin_time = time.time()

    # Create the searchlight object
    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)
    print("Setup searchlight inputs")
    print("Input data shape: " + str(data[0].shape))
    print("Input mask shape: " + str(mask.shape) + "\n")

    # Distribute the information to the searchlights (preparing it to run)
    sl.distribute([data], mask)

    # Execute searchlight
    print("Begin Searchlight\n")
    sl_result = sl.run_searchlight(calc_rsa, pool_size=pool_size)
    sl_result = sl_result.astype('double')
    sl_result[np.isnan(sl_result)] = 0
    print("End Searchlight\n")
    end_time = time.time()
    print('Total searchlight duration (including start up time): %.2f' % (end_time - begin_time))
    sl_nii = nib.Nifti1Image(sl_result, affine_mat)
    nib.save(sl_nii,
             "/content/rsatoolbox/isiktoolbox/data/searchlight/sl_sub_" + str(sub) + "_layer_" + layer + ".nii.gz")
    print("Saved Searchlight results to /content/rsatoolbox/isiktoolbox/data/searchlight/sl_sub_" + str(
        sub) + "_layer_" + layer + ".nii.gz")
