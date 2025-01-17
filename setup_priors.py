# this file contains functions required for
# setting up the mean and variance of different channels of GPs

import torch
from covariance_fns import covariance_function
from flags import (NUM_FRAMES,
                   BATCH_SIZE,
                   NDIM,
                   NUM_FEA,
                   FEA_DIM)


# returns inverse variance and determinant of latent priors
def setup_pz(num_fea, fea_dim, fea_priors, device="cuda"):

    sigmas, dets = [], []

    for f in range(num_fea):
        prior = fea_priors[f]

        if (len(prior.split('_')) == 2):
            p_type, H = prior.split('_')[0], float(prior.split('_')[1])

        else:
            p_type, H = prior, None

        for n in range(fea_dim):
            raw_sigma, det_ = covariance_function(p_type, NUM_FRAMES, H)
            sigma = torch.from_numpy(raw_sigma).float()
            det = torch.tensor(det_)
            sigmas.append(sigma)
            dets.append(det)

    sigma_p = torch.stack(sigmas)
    sigma_p_inv = torch.inverse(sigma_p).to(device)
    det_p = torch.stack(dets).to(device)

    return sigma_p_inv, det_p


# set mean of prior on latent space
def get_prior_mean(mean_fea_s, mean_fea_e, device="cuda"):

    mean = torch.zeros(BATCH_SIZE, NDIM, NUM_FRAMES).to(device)

    if (NUM_FEA > 5 or NUM_FEA < 1):
        raise Exception(
            'Mean not implemented for NUM_FEA = {}'.format(NUM_FEA)
            )

    for i in range(NUM_FRAMES):
        for f in range(NUM_FEA):

            if (mean_fea_s[f] is None and mean_fea_e[f] is None):
                mean[:, f * FEA_DIM:(f + 1) * FEA_DIM, i] = (
                    mean_fea_s[i]
                    + i * ((mean_fea_e[f]
                            - mean_fea_s[f]) / (NUM_FRAMES-1)))

    return mean
