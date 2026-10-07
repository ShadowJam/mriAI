**English** | [Русский](README.ru.md)
# mriAI

# Multiple Sclerosis Diagnosis from Brain MRI

An educational research project: a prototype system for analyzing brain MRI scans to perform binary classification of multiple sclerosis (MS).

> **Disclaimer:** This project was created for educational and research purposes. It is not a medical device, has not undergone clinical validation, and is not intended for diagnosis, treatment, or medical decision-making. Any model output must not be interpreted as a medical conclusion.

---

## About the Project

The goal of this project was to work through a full medical imaging pipeline: from reading DICOM files and preprocessing 3D MRI series to training a deep learning model for classifying the presence of multiple sclerosis.

The project models a realistic scenario where MRI scans come from different scanners and are provided as sets of DICOM slices. The main challenge is to convert this data into a format suitable for training and to train a 3D convolutional model.

## Task

- Input: a set of brain MRI DICOM scans for a single patient.
- Output: a binary label — whether the patient has multiple sclerosis.
- Key aspect: working with 3D series of slices rather than individual 2D images.

## Data

### Source and Structure

| Data source | Link | Data type | Number of records | Label type | Label classes | Class distribution, % | Publication date |
|---|---|---|---|---|---|---|---|
| GBUZ "NPKTS DiT DZM", mosmed.ai | https://mosmed.ai/datasets/aie21selftestmri/ | DICOM, MRI scans | 172 | Binary: with pathology / without pathology | 1/0 | 50/50 | 22.10.2022 |

Expected directory structure:


```
SCLEROSIS/
├── labels.xlsx
├── MRI/
│   └── <patient_id>/
│       └── <series_folder>/
│           └── *.dcm
├── dcm.json
└── weights.pth
```

## Model

### Architecture

The model is based on a 3D U-Net-like architecture, adapted for classifying 3D MRI volumes.

Main blocks:

- **DoubleConv** — two consecutive 3D convolutional layers `3 × 3 × 3` with BatchNorm3d and ReLU. Used as the base building block.
- **Down** — downscaling block: `MaxPool3d(2 × 2 × 2)` + `DoubleConv`. Reduces spatial resolution and increases the number of channels.
- **Up** — upscaling block: `Upsample / ConvTranspose3d` + skip-connection concatenation with the corresponding encoder level + `DoubleConv`. Restores spatial resolution.
- **OutConv** — output layer `1 × 1 × 1`, maps channels to the number of classes.

### Input and Output

- Model input: `[B, 3, 20, 200, 200]`
  - `B` — batch size;
  - `3` — channels (a single 3D volume is duplicated three times to match the architecture's expectations);
  - `20` — number of slices in the volume;
  - `200 × 200` — spatial resolution of a single slice.

- Model output in the current version: `[B, 1, 20, 200, 200]`
  - the model returns a 3D volume of probabilities after sigmoid.

### Loss Function and Optimizer

- Loss: `BCELoss` (binary cross-entropy).
- Optimizer: `Adam` with `lr = 0.001`.
- Training: up to `40` epochs with early stopping when `loss < 0.3` after epoch 10.

### Metrics

Validation and test compute:

- Accuracy
- Precision
- Recall
- F1-score

## Tech Stack

- Python
- PyTorch
- pydicom
- NumPy
- pandas
- scikit-image
- OpenCV
- scikit-learn
- Matplotlib

## What Did Not Work and Why

An honest summary of the research: no significant results were achieved. The reason is a combination of an unsuitable architecture and the complexity of the dataset.

### Model

The chosen 3D U-Net is a **segmentation** architecture: it was designed for per-pixel labeling and returns a volume of the same dimensionality as its input. For binary classification (one label per patient) it is overengineered and misapplied:

- the decoder and skip-connections are unnecessary when the answer is not an image;
- the model output `[B, 1, 20, 200, 200]` is a 3D volume of probabilities, not a patient label;
- there are technical issues in training: mismatched shapes of `outputs` and `labels` in `BCELoss`, unstable `squeeze()` without specifying the axis, duplication of a single 3D volume into 3 channels without adding new information, heuristic gamma correction instead of proper normalization.

### Dataset

- **Small size** — a typical problem for medical datasets on MS; it is not enough for 3D networks with millions of parameters.
- **Different scanners** — without harmonization (z-score per scanner, histogram matching, ComBat), the model learns to distinguish scanners rather than pathology.
- **Different protocols** (T1, T2, FLAIR, with and without contrast) — mixing them confuses the model.
- **Patient-level labels, not lesion-level labels** — a conceptual conflict with a segmentation architecture.

### How to Fix It

**Model:**
- replace U-Net with classification-oriented 3D networks;
- add `AdaptiveAvgPool3d(1)` + `Linear(features, 1)` on the output;
- use `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance;
- remove channel duplication — feed a single 3D volume.

**Data:**
- normalize intensities per-volume (z-score inside the brain mask) instead of gamma;
- resample to a fixed `spacing` via `SimpleITK` or MONAI transforms instead of blind cropping;
- add harmonization between scanners;
- separate protocols or encode them explicitly;
- add augmentations: random flip, rotation, intensity shift.

### Conclusion

The lack of results is not a model failure, but a consequence of an architecture unsuited to the classification task and a complex, heterogeneous dataset. The main lessons: architecture should be chosen for the task type, not the other way around; and for medical imaging, data preprocessing and harmonization matter no less than the model itself.
### Вывод

Отсутствие результатов — это не провал модели, а следствие неподходящей архитектуры под задачу классификации и сложного, неоднородного датасета. Основные уроки: архитектуру выбирают под тип задачи, а не наоборот; а для медицинских изображений предобработка и harmonization данных значат не меньше, чем сама модель.
