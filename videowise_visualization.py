# this file contains code that helps visualize the latent space encodings
# note: this is valid only when all features are kept 2-dimensional
# (i.e. FEA_DIM == 2)

import os
import torch
from itertools import cycle

import matplotlib.pyplot as plt
from flags import (BATCH_SIZE,
                   ENCODER_SAVE,
                   DECODER_SAVE,
                   NUM_FEA,
                   NUM_POINTS_VISUALIZATION,
                   CUDA)
from networks import Encoder, Decoder

from torch.utils.data import DataLoader
from utils import weights_init
from dataloader import load_dataset

import matplotlib
matplotlib.use('agg')


def choose_color(i):
    # size of this list should be same as number of videos to be plotted
    # (i.e. NUM_POINTS_VISUALIZATION)
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    # Add more colors here if you wish to plot more videos

    return colors[i]


if __name__ == "__main__":

    if not os.path.exists('./results/video_visualization/'):
        os.makedirs('./results/video_visualization/')

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

    if (CUDA and torch.cuda.is_available()):
        encoder.cuda()
        decoder.cuda()
    elif (CUDA and torch.backends.mps.is_available()):
        encoder = encoder.to("mps")
        decoder = decoder.to("mps")

    for i in range(NUM_FEA):

        fea_x = []
        fea_y = []

        for j in range(NUM_POINTS_VISUALIZATION):

            X_in = next(loader)
            X_in = X_in.float()

            if (CUDA and torch.cuda.is_available()):
                X_in = X_in.cuda()
            elif (CUDA and torch.backends.mps.is_available()):
                X_in = X_in.to("mps")

            X1, KL1, muL1, det_q1 = encoder(X_in)
            X1 = X1.data.cpu().numpy()

            fea_x.append(X1[:, :, 2*i])
            fea_y.append(X1[:, :, 2*i + 1])

        fig, ax = plt.subplots(1)

        for j in range(NUM_POINTS_VISUALIZATION):

            # plottings video encodings with different color for each video
            plt.scatter(fea_x[j],
                        fea_y[j],
                        marker='o',
                        c=choose_color(j),
                        cmap=plt.cm.get_cmap("jet", 10),
                        edgecolor='k')

        # may vary the limits of this figure depending on the
        # spread of each Gaussian process
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.savefig("./results/video_visualization/fea{}.png".format(str(i)))

        plt.close()
