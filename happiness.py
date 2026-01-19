import numpy as np

# Logistic function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Model: P(happy | AU6, AU12)
def happiness_probability(AU6, AU12, params):
    """
    AU6, AU12: binary (0 or 1)
    params: [β0, β6, β12, β6_12]
    """
    β0, β6, β12, β6_12 = params
    logit = β0 + β6*AU6 + β12*AU12 + β6_12*(AU6*AU12)
    return sigmoid(logit)

# Initial parameter estimates
params = [-2.0, 0.5, 1.5, 1.0]

# Test all four AU combinations
print("Model predictions:")
for AU6 in [0, 1]:
    for AU12 in [0, 1]:
        p = happiness_probability(AU6, AU12, params)
        decision = "HAPPY" if p > 0.5 else "NOT HAPPY"
        print(f"AU6={AU6}, AU12={AU12}: P={p:.3f} → {decision}")

# Validation against CK+ empirical data
def validate_model(params, dataset):
    """
    dataset: list of (AU6, AU12, true_emotion) tuples
    Returns: accuracy, confusion matrix metrics
    """
    correct = 0
    total = len(dataset)

    # Confusion matrix components
    true_positive = 0   # correctly classified happy
    false_positive = 0  # non-happy classified as happy
    false_negative = 0  # happy classified as non-happy
    true_negative = 0   # correctly classified non-happy

    for AU6, AU12, true_emotion in dataset:
        p = happiness_probability(AU6, AU12, params)
        predicted = "happy" if p > 0.5 else "not_happy"

        if true_emotion == "happy" and predicted == "happy":
            true_positive += 1
            correct += 1
        elif true_emotion == "happy" and predicted == "not_happy":
            false_negative += 1
        elif true_emotion != "happy" and predicted == "happy":
            false_positive += 1
        else:
            true_negative += 1
            correct += 1

    accuracy = correct / total

    # Target from Table 7: 100% for happiness
    print(f"\nValidation Results:")
    print(f"Overall Accuracy: {accuracy:.1%}")
    print(f"True Positives (happy→happy): {true_positive}")
    print(f"False Negatives (happy→not happy): {false_negative}")
    print(f"False Positives (not happy→happy): {false_positive}")
    print(f"True Negatives (not happy→not happy): {true_negative}")

    return accuracy

# Create synthetic CK+ dataset based on paper
# Assumption: all 69 happy faces have AU6+AU12
# Need to estimate non-happy AU patterns
dataset = []

# 69 happy expressions with AU6+AU12
for i in range(69):
    dataset.append((1, 1, "happy"))

# Estimate non-happy: some have AU12 only, most have neither
# (Need your help estimating these from paper)

# 258 non-happy: estimate AU patterns
# Conservative assumption: most non-happy lack AU6+AU12
# From Table 1: AU6=122 total, AU12=111 total
# If 69 are in happy faces, ~53 AU6 and ~42 AU12 in non-happy

# Estimate: ~10 non-happy have AU6+AU12 (false positive risk)
for i in range(10):
    dataset.append((1, 1, "not_happy"))

# ~40 have AU12 only (social smiles without genuine markers)
for i in range(40):
    dataset.append((0, 1, "not_happy"))

# ~208 have neither (remaining non-happy expressions)
for i in range(208):
    dataset.append((0, 0, "not_happy"))

# Run validation
accuracy = validate_model(params, dataset)

# Compare to Table 7 target: 100% for happiness
print(f"\nTarget (Table 7): 100%")
print(f"Your model: {accuracy:.1%}")
