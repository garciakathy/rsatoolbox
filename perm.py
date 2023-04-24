from scipy.stats import spearmanr
import pandas as pd
import numpy as np

evc_corr = []
ffa_corr = []
ppa_corr = []
loc_corr = []
sts_corr = []
eba_corr = []
psts_corr = []
asts_corr = []
mt_corr = []

rois = ['evc', 'ppa', 'ffa', 'loc', 'face', 'eba', 'psts', 'asts', 'mt']
subjects = range(1,5)
shuffle = 0

#model layers
layer1 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.0.npz')['arr_0']
layer2 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.3.npz')['arr_0']
layer3 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.6.npz')['arr_0']
layer4 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.8.npz')['arr_0']
layer5 = np.load('/content/rsatoolbox/isiktoolbox/data/rdms/AlexNet/features.10.npz')['arr_0']
layer_lower1 = layer1[np.tril_indices_from(layer1, k=-1)]
layer_lower2 = layer2[np.tril_indices_from(layer2, k=-1)]
layer_lower3 = layer3[np.tril_indices_from(layer3, k=-1)]
layer_lower4 = layer4[np.tril_indices_from(layer4, k=-1)]
layer_lower5 = layer5[np.tril_indices_from(layer5, k=-1)]

for i in range(0,5001):
  for roi in rois:
    for sub in subjects:
      sub = str(sub)
      # Loading Betas
      betas = np.load("isiktoolbox/data/RoiSubjectBetas/"+roi+"_betas_"+str(sub)+".npz")['arr_0']
      # Shuffle the videos
      if shuffle != 0:
        np.random.shuffle(betas)

      # Correlating pairwise across 200 videos
      df_beta = pd.DataFrame(betas)
      df_pearson = 1 - df_beta.corr(method='pearson')
      sub_rdm = df_pearson.to_numpy()

      # RSA
      lower = sub_rdm[np.tril_indices_from(sub_rdm, k=-1)]
      r1 = spearmanr(lower, layer_lower1)[0]
      r2 = spearmanr(lower, layer_lower2)[0]
      r3 = spearmanr(lower, layer_lower3)[0]
      r4 = spearmanr(lower, layer_lower4)[0]
      r5 = spearmanr(lower, layer_lower5)[0]

      # Adding to brain rdm (p, n x n)
      if roi == 'evc':
        evc_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'ffa':
        ffa_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'ppa':
        ppa_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'eba':
        eba_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'loc':
        loc_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'face':
        sts_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'psts':
        psts_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'asts':
        asts_corr.append([r1,r2,r3,r4,r5])
      elif roi == 'mt':
        mt_corr.append([r1,r2,r3,r4,r5])

  print(str(shuffle) + " / 5000 Done")
  shuffle += 1

# Save RDM's as zipped numpy arrays locally
brain_rdms_arr = np.array(evc_corr)
np.savez("isiktoolbox/data/shuffledSpearman/evc_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(ffa_corr)
np.savez("isiktoolbox/data/shuffledSpearman/ffa_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(ppa_corr)
np.savez("isiktoolbox/data/shuffledSpearman/ppa_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(loc_corr)
np.savez("isiktoolbox/data/shuffledSpearman/loc_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(sts_corr)
np.savez("isiktoolbox/data/shuffledSpearman/sts_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(eba_corr)
np.savez("isiktoolbox/data/shuffledSpearman/eba_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(psts_corr)
np.savez("isiktoolbox/data/shuffledSpearman/psts_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(asts_corr)
np.savez("isiktoolbox/data/shuffledSpearman/asts_correlations.npz", brain_rdms_arr)

brain_rdms_arr = np.array(mt_corr)
np.savez("isiktoolbox/data/shuffledSpearman/mt_correlations.npz", brain_rdms_arr)
print('Finished all RDMs')