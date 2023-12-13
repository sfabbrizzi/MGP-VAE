import os
import torch

from flags import (BATCH_SIZE,
                   ENCODER_SAVE,
                   DECODER_SAVE,
                   CUDA,
                   NUM_FEA,
                   FEA_DIM,
                   NUM_FRAMES,
                   NUM_INPUT_CHANNELS,
                   NDIM)
from networks import Encoder, Decoder
from dataloader import load_dataset
from itertools import cycle
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import weights_init

import matplotlib
matplotlib.use('agg')

if __name__ == "__main__":

    if not os.path.exists('./results/style_transfer_results'):
        os.makedirs('./results/style_transfer_results')

    dataset = load_dataset()
    loader = cycle(DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True))

    encoder = Encoder()
    encoder.apply(weights_init)

    decoder = Decoder()
    decoder.apply(weights_init)

    encoder.load_state_dict(torch.load(os.path.join('checkpoints/',
                                                    ENCODER_SAVE)))
    decoder.load_state_dict(torch.load(os.path.join('checkpoints/',
                                                    DECODER_SAVE)))

    encoder.eval()
    decoder.eval()

    video1 = next(loader).float()[0].unsqueeze(0)
    video2 = next(loader).float()[0].unsqueeze(0)

    if (CUDA and torch.cuda.is_available()):
        encoder.cuda()
        decoder.cuda()

        video1 = video1.cuda()
        video2 = video2.cuda()
    elif (CUDA and torch.backends.mps.is_available()):
        encoder = encoder.to("mps")
        decoder = decoder.to("mps")

        video1 = video1.to("mps")
        video2 = video2.to("mps")

    X1, KL1, muL1, det_q1 = encoder(video1)
    X2, KL2, muL2, det_q2 = encoder(video2)

    # save reconstructed images
    dec_v1 = decoder(X1)
    save_image(dec_v1.squeeze(0).transpose(2, 3),
               './results/style_transfer_results/recon_v1.png',
               nrow=NUM_FRAMES,
               normalize=True)

    dec_v2 = decoder(X2)
    save_image(dec_v2.squeeze(0).transpose(2, 3),
               './results/style_transfer_results/recon_v2.png',
               nrow=NUM_FRAMES,
               normalize=True)

    v1_feature = []
    v2_feature = []

    for i in range(NUM_FEA):

        v1_feature.append(X1[:, :, i*FEA_DIM:(i+1)*FEA_DIM])
        v2_feature.append(X2[:, :, i*FEA_DIM:(i+1)*FEA_DIM])

    for i in range(NUM_FEA):

        for j in range(NUM_FEA):

            # style transfer on video1
            v1_feature_transferred = torch.zeros(NUM_INPUT_CHANNELS,
                                                 NUM_FRAMES,
                                                 NDIM)

            if (CUDA and torch.cuda.is_available()):
                v1_feature_transferred = v1_feature_transferred.cuda()

            elif (CUDA and torch.backends.mps.is_available()):
                v1_feature_transferred = v1_feature_transferred.to("mps")

            if (j == i):
                v1_feature_transferred[:,
                                       :,
                                       i*FEA_DIM:(i+1)*FEA_DIM] = v2_feature[j]
            else:
                v1_feature_transferred[:,
                                       :,
                                       i*FEA_DIM:(i+1)*FEA_DIM] = v1_feature[j]

            v1_feature_transferred_dec = decoder(v1_feature_transferred)
            save_image(
                v1_feature_transferred_dec.squeeze(0).transpose(2, 3),
                ('./results/style_transfer_results/'
                 'v1_grid_feature{}_transferred.png').format(j),
                nrow=NUM_FRAMES, normalize=True)

            # style transfer on video2
            v2_feature_transferred = torch.zeros(NUM_INPUT_CHANNELS,
                                                 NUM_FRAMES,
                                                 NDIM)
            if (CUDA and torch.cuda.is_available()):
                v2_feature_transferred = v2_feature_transferred.cuda()

            elif (CUDA and torch.backends.mps.is_available()):
                v2_feature_transferred = v2_feature_transferred.to("mps")

            if (j == i):
                v2_feature_transferred[:,
                                       :,
                                       i*FEA_DIM:(i+1)*FEA_DIM] = v1_feature[j]
            else:
                v2_feature_transferred[:,
                                       :,
                                       i*FEA_DIM:(i+1)*FEA_DIM] = v2_feature[j]

            v2_feature_transferred_dec = decoder(v2_feature_transferred)
            save_image(v2_feature_transferred_dec.squeeze(0).transpose(2, 3),
                       ('./results/style_transfer_results/'
                        'v2_grid_feature{}_transferred.png').format(j),
                       nrow=NUM_FRAMES,
                       normalize=True)
