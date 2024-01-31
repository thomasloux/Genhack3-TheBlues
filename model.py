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
    latent_variable = noise[:, ...]  # choose the appropriate latent dimension of your model

    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    model = ...

    return model(latent_variable) # G(Z)
    # return model(latent_variable, scenario) # G(Z, x)




