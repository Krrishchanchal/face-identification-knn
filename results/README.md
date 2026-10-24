# Results Directory

The notebook will save reproducible experimental outputs here, including:

- Dataset and class-distribution figures
- Sample faces and Eigenfaces
- Hyperparameter-selection plots
- Confusion matrix
- Correct and incorrect prediction examples
- Machine-readable metrics

Model-training and evaluation results have not been generated yet.

## Dataset Exploration Outputs

- `lfw_identity_distribution.png` shows the 15 most represented identities in the complete dataset.
- `lfw_sample_faces.png` shows one deterministic sample from each of the 10 most represented identities.

## Balanced Dataset Outputs

- `balanced_identity_distribution.png` verifies that all five selected identities contain exactly 50 images.
- `balanced_sample_faces.png` shows five reproducibly sampled faces from each selected identity, illustrating variations in pose, expression, and illumination.

## Data-Split Output

- `train_test_distribution.png` verifies the stratified 80:20 split: every identity contributes 40 training images and 10 held-out testing images.

