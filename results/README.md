# Results Directory

The notebook will save reproducible experimental outputs here, including:

- Dataset and class-distribution figures
- Sample faces and Eigenfaces
- Hyperparameter-selection plots
- Confusion matrix
- Correct and incorrect prediction examples
- Machine-readable metrics

The directory currently contains the dataset analysis, model-selection results, and held-out outputs for the 50-images-per-identity pilot experiment. These files are retained as a transparent baseline while a larger experiment is developed.

## Dataset Exploration Outputs

- `lfw_identity_distribution.png` shows the 15 most represented identities in the complete dataset.
- `lfw_sample_faces.png` shows one deterministic sample from each of the 10 most represented identities.

## Balanced Dataset Outputs

- `balanced_identity_distribution.png` verifies that all five selected identities contain exactly 50 images.
- `balanced_sample_faces.png` shows five reproducibly sampled faces from each selected identity, illustrating variations in pose, expression, and illumination.

## Data-Split Output

- `train_test_distribution.png` verifies the stratified 80:20 split: every identity contributes 40 training images and 10 held-out testing images.

## PCA and Eigenface Outputs

- `pca_explained_variance.png` shows how cumulative training-set variance increases across the first 100 PCA components; together they retain 95.92%.
- `eigenfaces.png` visualises the first 15 principal components reshaped to the original 62 x 47 image dimensions.

## Model-Selection Outputs

- `cross_validation_results.csv` records the mean accuracy, accuracy standard deviation, and rank for all 50 PCA and k-NN parameter combinations.
- `parameter_selection.png` compares five-fold validation accuracy across PCA dimensions, neighbour counts, and voting methods. The selected configuration uses 40 components, `k=9`, and distance weighting, with 60.50% mean validation accuracy.

## Pilot Evaluation Outputs

- `final_metrics.json` stores the pilot's 52.00% test accuracy, prediction counts, validation accuracy, and selected parameters. The filename predates the decision to retain this run as a pilot baseline.
- `classification_report.csv` stores precision, recall, F1-score, and support for each identity.
- `test_predictions.csv` records the actual identity, predicted identity, confidence, and correctness for every held-out image.
- `confusion_pairs.csv` ranks directed identity-confusion errors by frequency.
- `confusion_matrix.png` presents the pilot's complete five-class held-out confusion matrix.
- `sample_predictions.png` shows six correct and six incorrect held-out predictions with model confidence.

## Expanded Dataset Outputs

- `expanded_dataset_manifest.csv` records every selected LFW source index, identity, split, and whether the image belonged to the pilot or newly sampled pool.
- `expanded_dataset_summary.csv` records the transition from 50 pilot images to 87 training and 22 fresh test images per identity.
- `expanded_split_distribution.png` verifies the balanced 435-image training set and 110-image fresh final test set.

## Expanded Model-Selection Outputs

- `expanded_cross_validation_results.csv` records mean accuracy, standard deviation, and rank for all 96 expanded-training configurations.
- `expanded_model_selection.json` records the selected parameters, 61.15% mean validation accuracy, 3.81 percentage-point standard deviation, and confirmation that zero test images were used during selection.
- `expanded_parameter_selection.png` compares the five-fold validation accuracy across six PCA dimensions, eight neighbour counts, and two voting methods. The selected configuration uses 40 components, `k=15`, and distance weighting.

