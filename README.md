# Face Identification Using PCA and k-Nearest Neighbours

A closed-set face-identification system that recognises one person from a gallery of five known identities. The project represents facial images with PCA-based Eigenfaces and classifies them using k-Nearest Neighbours (k-NN).

## Overview

Face identification is a one-to-many matching problem: a query face is compared with several known identities, and the system returns the closest match. This project builds a compact and interpretable identification pipeline using classical machine-learning techniques rather than a deep neural network.

The system uses the Labeled Faces in the Wild (LFW) dataset and evaluates performance on a balanced five-person subset.

## Project Goals

- Build a reproducible face-identification pipeline.
- Select five identities using a documented and repeatable rule.
- Maintain an equal number of images for every identity.
- Extract compact facial features using PCA/Eigenfaces.
- Select the number of PCA components and the value of `k` through cross-validation.
- Evaluate the final model on an untouched test set.
- Analyse correct predictions, misclassifications, and commonly confused identities.

## Methodology

```text
LFW face images
       |
       v
Balanced five-identity subset
       |
       v
Stratified train-test split
       |
       v
PCA / Eigenface feature extraction
       |
       v
k-NN classification
       |
       v
Evaluation and error analysis
```

The planned workflow is:

1. Load LFW through Scikit-learn.
2. Select five identities with sufficient images.
3. Sample the same number of images for each identity.
4. Create a stratified training and testing split.
5. Fit PCA using training data only to prevent data leakage.
6. Transform each image into an Eigenface feature vector.
7. Select model parameters using cross-validation on the training set.
8. Evaluate the selected model once on the held-out test set.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- face_identification.ipynb
|-- report/
|   `-- final_report.pdf
|-- results/
|   |-- confusion_matrix.png
|   |-- eigenfaces.png
|   |-- parameter_selection.png
|   `-- sample_predictions.png
`-- src/
    `-- face_identification.py
```

The notebook will contain the complete exploratory and experimental workflow. Reusable functions may also be placed in `src/` as the implementation develops.

## Installation

Python 3.10 or later is recommended.

```powershell
git clone https://github.com/Krrishchanchal/face-identification-knn.git
cd face-identification-knn

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter Notebook with:

```powershell
jupyter notebook
```

Then open `face_identification.ipynb` and run the cells in order.

The LFW dataset will be downloaded automatically by Scikit-learn during the first run. Downloaded data is excluded from Git.

## Experimental Design

The initial experiment uses:

- 5 known identities
- A balanced number of images per identity
- An 80:20 stratified train-test split
- PCA/Eigenfaces for dimensionality reduction
- k-NN for classification
- Cross-validation for parameter selection

The test set remains untouched until model selection is complete. This provides a more reliable estimate of performance on unseen images of the five known people.

## Evaluation

The system will be evaluated using:

- Classification accuracy
- Precision, recall, and F1-score for each identity
- Confusion matrix
- Cross-validation performance
- Correct and incorrect prediction examples
- Visual analysis of frequently confused identities

Final metrics and generated figures will be added after the complete experiment has been run.

## Limitations

This is a closed-set identification system: every test image is assumed to belong to one of the five known identities. Standard k-NN will always return one of those identities, even when the query contains an unknown person. Supporting unknown identities would require an open-set evaluation protocol and a validated rejection threshold.

Performance may also be affected by changes in pose, facial expression, illumination, occlusion, and image quality.

## Future Improvements

- Add unknown-person rejection.
- Compare uniform and distance-weighted k-NN.
- Evaluate more identities and more challenging test images.
- Add automatic face detection and alignment.
- Compare Eigenfaces with HOG features and deep face embeddings.
- Evaluate robustness on CCTV-like images.

## References

Dataset and methodology references will be added as the implementation is completed.
