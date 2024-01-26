import matplotlib.pyplot as plt
import numpy as np


def visualize_model(model):
    params = list(model.parameters())
    flat_layer_0 = np.array(
        list(params[0].detach().numpy().T) + [list(params[1].detach().numpy())]
    ).T
    flat_layer_1 = np.array(
        list(params[2].detach().numpy().T) + [list(params[3].detach().numpy())]
    ).T
    flat_layer_2 = np.array(list(params[4].detach().numpy().T))

    for i, im_data in enumerate([flat_layer_0, flat_layer_1, flat_layer_2]):
        plt.subplot(131 + i)
        plt.imshow(im_data, aspect='auto', vmin=-0.2, vmax=0.2)
        plt.axis('off')
    plt.colorbar()
    plt.show()

