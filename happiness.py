import numpy as np

def happiness_probability(AU6, AU12, params):
    beta0, beta6, beta12, beta6_12 = params
    logit = beta0 + beta6 + beta6*AU6 + beta12*AU12 + beta6_12*(AU6*AU12)
    return 1 / (1 + np.exp(-logit))

# paramater estimates
params = [-2.0, 0.5,  1.5, 1.0]

# test all combinations
for AU6 in [0, 1]:
    for AU12 in [0, 1]:
        p = happiness_probability(AU6, AU12, params)
        print(f"AU6={AU6}, AU12={AU12}: P={p:.3f}")
