# -*- coding: utf-8 -*-
"""
AImri.py
========

Multiple sclerosis diagnosis prototype based on brain MRI.

An educational research project exploring binary classification of
multiple sclerosis (MS) from 3D brain MRI series in DICOM format.

.. warning::
    This is an educational project. It is not a medical device and
    is not intended for clinical diagnosis, treatment, or any
    medical decision-making.

Overview
--------
The pipeline reads DICOM series per patient, selects the axial series,
preprocesses slices, assembles a fixed-size 3D volume, caches it to
JSON Lines, and trains a 3D convolutional model on the cached data.

The file is kept as a research artifact. It intentionally preserves
commented-out experiments to document what was tried and why certain
approaches were abandoned.

File map
--------
1. Imports
2. Configuration
       Paths, volume size, and slice count constants.
3. Device selection
       Picks CUDA if available, otherwise CPU.
4. JSON Lines helpers
       ``json_save`` / ``json_load`` for caching preprocessed volumes.
5. Padding helper
       ``pad_slice`` builds a zero-filled slice.
6. Early exploratory DICOM inspection
       Metadata lookups on single DICOM files. Commented out.
7. Early preprocessing experiments
       Normalization, filtering, and visualization attempts.
       Commented out, kept for reference.
8. Padding helpers for 3D volumes
       ``max_angle_slice_len`` and older ``pad_image``.
       Superseded by hardcoded ``max_slice``.
9. DICOM reading and preprocessing
       ``read_patient`` — main per-patient preprocessing routine.
10. Older per-series stacking approach
        Multi-series stacking into a 4D tensor. Abandoned.
11. Dataset loading and caching
        ``load_data`` walks over labels and writes JSON Lines.
12. Dataset preparation
        ``split_data_labels`` and ``prepare_dataloaders``.
13. Evaluation
        ``evaluate_model`` computes accuracy, precision, recall, F1.
14. Model: 3D U-Net
        ``DoubleConv``, ``Down``, ``Up``, ``OutConv``, ``UNet``.
15. Alternative model experiments
        DenseNet-based, custom 3D CNN, and early multi-view models.
        Commented out, kept for reference.
16. Sanity check for the custom 3D model
        Shape-only smoke test. Commented out.
17. Training
        ``train`` — full training and evaluation loop.
18. Entry point
        Reads labels, preprocesses data, runs training.

Known limitations
-----------------
- The 3D U-Net is a segmentation architecture used here for
  classification. The classification head (``UNet.fcl``) is defined
  but not used in ``forward``.
- ``UNet.forward`` returns a 3D volume of probabilities, not a scalar.
- ``BCELoss`` receives tensors of mismatched shapes (see training loop).
- Input channel is duplicated three times to fit the backbone.
- Slice count and spatial size are fixed (20 x 200 x 200).
- The volume is cropped blindly around the center; lesion slices
  outside that window are lost.

Data layout
-----------
Expected directory structure::

    SCLEROSIS/
    ├── labels.xlsx
    ├── MRI/
    │   └── <patient_id>/
    │       └── <series_folder>/
    │           └── *.dcm
    ├── dcm.json
    └── weights.pth

``labels.xlsx`` contains a patient identifier and a binary label
(0 — no MS, 1 — MS).
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json

import numpy as np
import pandas as pd
import pydicom
import cv2

from skimage.transform import resize

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.models as models
import torchvision.transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
labels_path  = 'D:\\AImri\\SCLEROSIS\\labels.xlsx'
data_path    = 'D:\\AImri\\SCLEROSIS\\MRI'
json_path    = 'D:\\AImri\\SCLEROSIS\\dcm.json'
weights_path = 'D:\\AImri\\SCLEROSIS\\weights.pth'

max_slice = 20    # fixed number of slices per patient volume
img_size  = 200   # spatial size of each slice (img_size x img_size)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Training on GPU")
else:
    device = torch.device("cpu")
    print("Training on CPU")


# ---------------------------------------------------------------------------
# JSON Lines helpers for caching preprocessed data
# ---------------------------------------------------------------------------
def json_save(data, json_file_path):
    """Append one record to a JSON Lines file."""
    with open(json_file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')


def json_load(json_file_path):
    """Load all records from a JSON Lines file."""
    data = []
    with open(json_file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------
def pad_slice():
    """Return an empty (zero) slice of shape (img_size, img_size)."""
    return [[0] * img_size for _ in range(img_size)]


# ---------------------------------------------------------------------------
# Early exploratory DICOM inspection
# ---------------------------------------------------------------------------
# This block was used to look at a single DICOM file and inspect its metadata
# before building the pipeline. Kept here as a reference for what was examined.

# ph = 'D:\\AImri\\SCLEROSIS\\MRI\\1.2.643.5.1.13.13.12.2.77.8252.12071207070510001104140008021400\\'
# ph = 'D:\\AImri\\SCLEROSIS\\MRI\\1.2.643.5.1.13.13.12.2.77.8252.07000803110704080314101201140507\\'
# phhlst = os.listdir(ph)
# phh1 = os.path.join(ph, phhlst[1])
# phh2 = os.path.join(ph, phhlst[3])
# dcm_fl1 = pydicom.dcmread(os.path.join(phh1, os.listdir(phh1)[4]))
# dcm_fl2 = pydicom.dcmread(os.path.join(phh2, os.listdir(phh2)[0]))
# keys = dcm_fl1.keys()
# for i in keys:
#     print(dcm_fl1.get(i))
#     print(dcm_fl2.get(i))


# ---------------------------------------------------------------------------
# Early preprocessing experiments (kept for reference)
# ---------------------------------------------------------------------------
# Several normalization and filtering ideas were tried before settling on
# gamma correction + resize. These are left here to document the exploration.

# dicom_image = dcm_fl1.pixel_array
# plt.imshow(dicom_image, cmap='bone')
# plt.show()
# dicom_image = (dicom_image - np.min(dicom_image)) / (np.max(dicom_image) - np.min(dicom_image))
# dicom_image = np.power(dicom_image / 255, 0.6) * 255

# mean_intensity = np.mean(dicom_image)
# std_intensity = np.std(dicom_image)
# z_score_normalized_image = (dicom_image - mean_intensity) / std_intensity
# z_score_normalized_image = cv2.normalize(z_score_normalized_image, None, 1, 255, cv2.NORM_MINMAX)
# plt.imshow(dicom_image, cmap='bone')
# plt.show()

# dicom_image = resize(dicom_image, (200, 200), anti_aliasing=True)
# plt.imshow(dicom_image, cmap='bone')
# plt.show()

# dicom_image = np.array(dicom_image, dtype=np.uint8)
# dicom_image = cv2.medianBlur(dicom_image, 3)
# dicom_image = cv2.createCLAHE(clipLimit=0.0001, tileGridSize=(20,20)).apply(dicom_image)

# for i in range(len(os.listdir(phh1))):
#     print(pydicom.dcmread(os.path.join(phh1, os.listdir(phh1)[i])).InstanceNumber)

# Notes made during exploration:
# - sort slices by InstanceNumber
# - keep only one orientation [L, P]
# - consider averaging over color channels


# ---------------------------------------------------------------------------
# Padding helpers for 3D volumes (older approach, kept for reference)
# ---------------------------------------------------------------------------
def max_angle_slice_len(data_directory, df_labels):
    """
    Compute maximum series count and maximum slice count across all patients.

    This was used to pick the padding size before it was hardcoded.
    Kept for reference.
    """
    mx_angle = 0
    mx_slice = 0
    for patient in df_labels.itertuples():
        patient_folder_path = os.path.join(data_directory, str(patient[1]))
        if os.path.isdir(patient_folder_path):
            if mx_angle < len(os.listdir(patient_folder_path)):
                mx_angle = len(os.listdir(patient_folder_path))
            for side_folder in os.listdir(patient_folder_path):
                side_folder_path = os.path.join(patient_folder_path, side_folder)
                if mx_slice < len(os.listdir(side_folder_path)):
                    mx_slice = len(os.listdir(side_folder_path))
    return mx_angle, mx_slice


# def pad_image(nslice):
#     pimg = []
#     for i in range(nslice):
#         pimg.append(pad_slice())
#     pimg, _ = stack_dicom_images(pimg, 'padding_image')
#     return pimg

# max_angle, max_slice = max_angle_slice_len(data_path, df_xlabels)
# max_angle = 10


# ---------------------------------------------------------------------------
# DICOM reading and preprocessing
# ---------------------------------------------------------------------------
def read_patient(patient_folder_path):
    """
    Read and preprocess all DICOM series of a single patient.

    Selects the axial series (PatientOrientation == ['L', 'P']),
    sorts slices by InstanceNumber, applies gamma correction and resizing,
    then pads or crops the volume to a fixed number of slices.

    Returns:
        patient_array (list): 3D volume of shape (max_slice, img_size, img_size).
        res (bool): True if a valid series was found and processed.
    """
    patient_array = []
    res = False

    for serie in os.listdir(patient_folder_path):
        serie_path = os.path.join(patient_folder_path, serie)
        if not os.path.isdir(serie_path):
            continue

        seriedir = os.listdir(serie_path)

        # Skip series that contain only one file
        if len(seriedir) == 1:
            return patient_array, res

        # Check series orientation from the first DICOM file
        dicom_chk = pydicom.dcmread(os.path.join(serie_path, seriedir[0]))
        try:
            ptor = dicom_chk.PatientOrientation
        except Exception:
            print(f"No Patient Orientation; file path: {serie_path}")
            break

        # Only axial series are used
        if ptor != ['L', 'P']:
            continue

        # Find the maximum InstanceNumber to size the volume
        max_rnum = 0
        for dcm in seriedir:
            dicom_file_path = os.path.join(serie_path, dcm)
            if os.path.isfile(dicom_file_path) and dcm.lower().endswith('.dcm'):
                try:
                    dicom_image = pydicom.dcmread(dicom_file_path)
                    if max_rnum < dicom_image.InstanceNumber:
                        max_rnum = dicom_image.InstanceNumber
                except Exception:
                    pass

        # Initialize volume with empty slices
        patient_array = [pad_slice() for _ in range(max_rnum)]

        # Read and preprocess each slice of the series
        for dcm in seriedir:
            dicom_file_path = os.path.join(serie_path, dcm)
            if not (os.path.isfile(dicom_file_path) and dcm.lower().endswith('.dcm')):
                continue

            try:
                dicom_image = pydicom.dcmread(dicom_file_path)
            except Exception as e:
                print(f"Reading file error {dicom_file_path}: {e}")
                continue

            insnum = dicom_image.InstanceNumber
            try:
                dicom_image = dicom_image.pixel_array

                # Gamma correction
                # dicom_image = np.power(dicom_image / 255, 0.6) * 255
                dicom_image = np.maximum(dicom_image / 255, 0) ** 0.6 * 255

                # Resize with anti-aliasing
                dicom_image = resize(dicom_image, (img_size, img_size), anti_aliasing=True)

                # Place slice into the volume by InstanceNumber
                patient_array[insnum - 1] = dicom_image.tolist()
                res = True
            except Exception as e:
                print(f"Preprocess file error {dicom_file_path}: {e}")

        break  # process only the first matching axial series

    # Pad or crop volume to a fixed number of slices
    if res:
        pd_slice = pad_slice()
        if len(patient_array) < max_slice:
            for _ in range(max_slice - len(patient_array)):
                patient_array.append(pd_slice)
        elif len(patient_array) > max_slice:
            center = len(patient_array) // 2
            patient_array = patient_array[center - max_slice // 2 : center + max_slice // 2]

    return patient_array, res


# ---------------------------------------------------------------------------
# Older per-series stacking approach (kept for reference)
# ---------------------------------------------------------------------------
# Before moving to a single axial series, the plan was to stack multiple
# series per patient into a 4D tensor. This approach was abandoned.

# def stack_dicom_images(dicom_images, side_folder_path):
#     try:
#         stacked_image = np.stack(dicom_images, axis=0)
#         stacked_image = stacked_image.tolist()
#         return stacked_image, True
#     except ValueError as ve:
#         print(f"Stacking images error {side_folder_path}: {ve}")
#         return None, False

# def read_patient_multi_series(patient_folder_path):
#     patient_images = []
#     # loop on side folders inside patient folder
#     for side_folder in os.listdir(patient_folder_path):
#         side_folder_path = os.path.join(patient_folder_path, side_folder)
#         if os.path.isdir(side_folder_path):
#             dicom_images, res_dcm_img_open = read_dicom_images(side_folder_path)
#             if res_dcm_img_open:
#                 stacked_image, res_img_stacking = stack_dicom_images(dicom_images, side_folder_path)
#                 if res_img_stacking:
#                     patient_images.append(stacked_image)
#     res = not patient_images
#     if bool(patient_images):
#         if len(patient_images) < max_angle:
#             for i in range(max_angle - len(patient_images)):
#                 patient_images.append(pad_image(max_slice))
#         if len(patient_images) > max_angle:
#             patient_images = patient_images[:10]
#     return patient_images, res


# ---------------------------------------------------------------------------
# Dataset loading and caching
# ---------------------------------------------------------------------------
def load_data(data_directory, df_labels):
    """
    Walk over patients listed in df_labels, preprocess their DICOM series
    and store (volume, label) pairs in a JSON Lines file.
    """
    labels_len = df_labels.shape[0]
    cnt = 1
    print(f"Data processing has started. {labels_len} labels found.")

    for patient in df_labels.itertuples():
        patient_folder_path = os.path.join(data_directory, str(patient[1]))
        if not os.path.isdir(patient_folder_path):
            continue

        patient_images, is_valid = read_patient(patient_folder_path)
        if is_valid:
            record = (patient_images, int(patient[2]))
            json_save(record, json_path)
            print(f"{cnt}/{labels_len} data processed.")
            cnt += 1
        else:
            labels_len -= 1
            print(f"Data with ID: {str(patient[1])} was skipped")

    print("Data processing has finished.")


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def split_data_labels(data_with_labels):
    """Split a list of (volume, label) pairs into X and y."""
    x, y = [], []
    for volume, label in data_with_labels:
        x.append(volume)
        y.append(label)
    return x, y


def prepare_dataloaders(batch_size=8):
    """
    Load cached data, split into train/val/test, and wrap in DataLoaders.

    Returns:
        train_dataloader, val_dataloader, test_dataloader
    """
    dt = json_load(json_path)
    x, y = split_data_labels(dt)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42
    )

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset   = TensorDataset(torch.tensor(x_val),   torch.tensor(y_val))
    test_dataset  = TensorDataset(torch.tensor(x_test),  torch.tensor(y_test))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader):
    """Compute accuracy, precision, recall and F1 on a dataloader."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # Note: input channel expansion was disabled here
            # because the model was later fed 3-channel input from the loop.
            # inputs = inputs.unsqueeze(1)
            # inputs = torch.cat((inputs, inputs, inputs), 1)

            outputs = model(inputs)
            outputs = outputs.squeeze()

            # Alternative (argmax over classes) — not used for binary task:
            # _, predicted = torch.max(outputs, 0)

            predicts = [round(v) for v in outputs.tolist()]
            predictions.extend(predicts)
            targets.extend(labels.tolist())

    # Alternative (explicit threshold) — replaced by round():
    # predictions = [1 if pred > 0.5 else 0 for pred in predictions]

    accuracy  = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall    = recall_score(targets, predictions)
    f1        = f1_score(targets, predictions)

    return accuracy, precision, recall, f1


# ---------------------------------------------------------------------------
# Model: 3D U-Net (encoder / decoder with skip connections)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two 3D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool3d + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample / ConvTranspose3d + skip-connection + DoubleConv.

    Marked as needing rework in the original code. The pad logic below
    handles cases where upsampled tensor does not exactly match the
    encoder tensor shape (odd dimensions after pooling).
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Align spatial dimensions with the encoder branch
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2,
            diff_z // 2, diff_z - diff_z // 2,
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1x1 convolution that maps to n_classes channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    3D U-Net-like network used as a binary classifier (prototype).

    Note: the classification head (self.fcl) is present but not used in
    forward(). The model returns a 3D volume of probabilities instead of
    a scalar per patient. See README for discussion.
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Classification head (currently unused)
        self.fcl  = nn.Linear(200, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # Debug prints were used to trace tensor shapes at each step:
        # print(f'x {x.shape}')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.sigm(self.outc(x))

        # Classification head was tried here but disabled:
        # out = self.sigm(self.fcl(logits))
        return logits


# ---------------------------------------------------------------------------
# Alternative model experiments (did not work, kept for reference)
# ---------------------------------------------------------------------------
# Several alternative architectures were tried before settling on 3D U-Net.
# None of them trained successfully on this dataset. Kept as a record of
# what was attempted.

# --- 2D DenseNet121 as a backbone ---
# Did not converge in practice.
# def cnn_model(pretrained=True):
#     model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
#     model = model.to(device)
#     return model

# --- Pretrained DenseNet wrapped with a per-slice classifier ---
# Idea: run a 2D backbone on each slice, then aggregate per-slice outputs.
# Abandoned because of unstable training and unclear slice aggregation.
# class PTCustomModel(nn.Module):
#     def __init__(self, pretrained_model):
#         super(PTCustomModel, self).__init__()
#         self.feat = nn.Sequential(*list(pretrained_model.features))
#         self.inconv1 = nn.Conv2d(1, 3, kernel_size=(5, 5), stride=1, bias=False)
#         self.outpool = nn.MaxPool2d(kernel_size=(3,3))
#         self.flatten = nn.Flatten()
#         self.relu = nn.ReLU(inplace=True)
#         self.fcl1 = nn.Linear(4096, 2048)
#         self.fcl2 = nn.Linear(2048, 1024)
#         self.fcl3 = nn.Linear(1024, 512)
#         self.fcl4 = nn.Linear(512, 256)
#         self.fcl5 = nn.Linear(256, 1)
#         self.fcl6 = nn.Linear(20, 1)
#         self.sigm = nn.Sigmoid()
#
#     def forward(self, x):
#         outputs = []
#         for i in range(x.size(1)):
#             out = x[:, i, :, :]
#             out = out.unsqueeze(1)
#             out = self.inconv1(out)
#             out = self.feat(out)
#             out = self.outpool(out)
#             out = self.flatten(out)
#             out = self.relu(self.fcl1(out))
#             out = self.relu(self.fcl2(out))
#             out = self.relu(self.fcl3(out))
#             out = self.relu(self.fcl4(out))
#             out = self.sigm(self.fcl5(out))
#             outputs.append(out)
#         outputs = torch.cat(outputs, dim=1)
#         outputs = outputs.view(outputs.size(0), -1)
#         outputs = self.sigm(self.fcl6(outputs))
#         return outputs

# --- Fully custom 3D CNN classifier ---
# This was a "clean" classifier: 3D convs + global pooling + FC. Trained
# poorly on the small dataset. Kept as reference for what a simpler
# classifier looked like.
# class CustomModel(nn.Module):
#     def __init__(self):
#         super(CustomModel, self).__init__()
#
#         self.conv0 = nn.Conv3d(3, 64, kernel_size=(5,7,7), padding=1, stride=1, bias=False)
#         self.norm0 = nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.pool0 = nn.MaxPool3d(kernel_size=(2,2,2), padding=1)
#         self.dropout = nn.Dropout(p=0.2)
#
#         self.conv1 = nn.Conv3d(64, 128, kernel_size=(3,5,5), padding=1)
#         self.norm1 = nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.pool1 = nn.MaxPool3d(kernel_size=(1,3,3))
#
#         self.conv2 = nn.Conv3d(128, 256, kernel_size=(3,5,5), padding=1)
#         self.norm2 = nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.pool2 = nn.MaxPool3d(kernel_size=(1,3,3))
#
#         self.conv3 = nn.Conv3d(256, 512, kernel_size=(3,3,3), padding=1)
#         self.norm3 = nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.pool3 = nn.MaxPool3d(kernel_size=(3,3,3))
#
#         self.conv4 = nn.Conv3d(512, 1024, kernel_size=(3,3,3), padding=1)
#         self.norm4 = nn.BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.pool4 = nn.MaxPool3d(kernel_size=(3,3,3))
#
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(1024, 256)
#         self.fc2 = nn.Linear(256, 1)
#         self.sigm = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.relu(self.conv0(x))
#         x = self.norm0(x)
#         x = self.pool0(x)
#         x = self.dropout(x)
#
#         x = self.relu(self.conv1(x))
#         x = self.norm1(x)
#         x = self.pool1(x)
#         x = self.dropout(x)
#
#         x = self.relu(self.conv2(x))
#         x = self.norm2(x)
#         x = self.pool2(x)
#         x = self.dropout(x)
#
#         x = self.relu(self.conv3(x))
#         x = self.norm3(x)
#         x = self.pool3(x)
#         x = self.dropout(x)
#
#         x = self.relu(self.conv4(x))
#         x = self.norm4(x)
#         x = self.pool4(x)
#         x = self.dropout(x)
#
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.sigm(self.fc2(x))
#         return x

# --- First attempt: DenseNet121 shared across views ---
# Something went wrong here; the approach was later split into the
# versions above. Kept as the earliest artifact.
# def cnn_model(pretrained=True):
#     model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
#     model = model.to(device)
#     return model
#
# pretrained_model = cnn_model(pretrained=True)
#         self.features = pretrained_model.features
#         self.features[0] = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.conv2 = nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1)
#         self.fc2 = nn.Linear(20, 2)
#     def forward(self, x):
#         conv_outputs = []
#         for i in range(x.size(1)):
#             view_data = x[:, i, :, :, :]
#             conv_output = self.features(view_data)
#             conv_output = self.conv2(conv_output)
#             conv_outputs.append(conv_output)
#         combined_output = torch.cat(conv_outputs, dim=1)
#         combined_output = combined_output.view(combined_output.size(0), -1)
#         output = self.fc2(combined_output)
#         return output


# ---------------------------------------------------------------------------
# Sanity check for the custom 3D model (shape only)
# ---------------------------------------------------------------------------
# model = CustomModel()
# inp = torch.randn(8, 3, 20, 200, 200)
# output = model(inp)
# print(output.shape)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    """Full training pipeline: load data, build model, train, evaluate."""
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(batch_size=8)

    model = UNet(3, 1).to(device)
    print(model)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print("Weights loaded from:", weights_path)
    else:
        print("Weights file not found. Training from scratch.")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss, train_accuracy, val_accuracy_history = [], [], []
    epochs = 40

    for epoch in range(epochs):
        model.train()
        cur_loss, cur_acc = 0.0, 0.0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # Duplicate single channel to 3 channels to fit the backbone
            inputs = inputs.unsqueeze(1)
            inputs = torch.cat((inputs, inputs, inputs), 1)

            # Sanity check for NaNs in inputs:
            # print(torch.isnan(inputs).any())

            optimizer.zero_grad()
            outputs = model(inputs)

            # Visualization of the U-Net output for debugging:
            # plt.imshow(outputs[0][0][0].tolist(), cmap='bone')
            # plt.show()

            outputs = outputs.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Note: this per-batch accuracy is approximate.
            # It was added late, mainly for quick monitoring.
            cur_loss = loss.item()
            predicts = [round(v) for v in outputs.tolist()]
            cur_acc = accuracy_score(labels.tolist(), predicts)
            train_loss.append(cur_loss)
            train_accuracy.append(cur_acc)

        val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {cur_loss:.4f}, "
              f"Val Accuracy: {val_acc:.4f}, Val Precision: {val_prec:.4f}, "
              f"Val Recall: {val_rec:.4f}")
        val_accuracy_history.append(val_acc)
        print(f"Train Accuracy: {cur_acc:.4f}")

        if epoch > 10 and cur_loss < 0.3:
            torch.save(model.state_dict(), weights_path)
            break

    test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_dataloader)
    print(f"Test Accuracy: {test_acc:.4f}, Test Precision: {test_prec:.4f}, "
          f"Test Recall: {test_rec:.4f}")

    torch.save(model.state_dict(), weights_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Read labels
    df_xlabels = pd.read_excel(labels_path)

    # 2. Preprocess DICOM data and cache to JSON Lines
    load_data(data_path, df_xlabels)

    # 3. Train and evaluate
    train()