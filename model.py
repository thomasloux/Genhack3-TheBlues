#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z,x |-> G_\theta(Z,x)
############################################################################

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>

import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128, momentum=0.8),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, output_dim),
            nn.LeakyReLU(0)
        )

    def forward(self, x):
        return self.model(x)


def generative_model(noise, scenario):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    scenario: ndarray with shape (n_samples, n_scenarios=9)
        input categorical variable of the conditional generative model
    """
    # See below an example
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = np.argmax(scenario)
    latent_dim = [40, 30, 30, 30, 30, 50, 30, 30, 30]
    mins = torch.load('parameters/mins.pth')
    maxs = torch.load('parameters/maxs.pth')
    latent_variable = noise[:, :latent_dim[label]]  # choose the appropriate latent dimension of your model
    
    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    model = Generator(latent_dim, output_dim=4).to(device)
    model.load_state_dict(torch.load(f'parameters/generator_model_{label}.pth'), map_location=device)
    generated_data = model(torch.Tensor(latent_variable))
    generated_data = generated_data * (maxs[label]) + mins[label]
    return generated_data.detach().numpy() # G(Z, x)




