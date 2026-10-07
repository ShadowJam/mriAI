**English** | [Русский](README.ru.md)
# mriAI

# Multiple Sclerosis Diagnosis from Brain MRI

# EN

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

[English](README.md) | **Русский**
# mriAI 

# Система диагностики рассеянного склероза по МРТ головного мозга

Учебный исследовательский проект: прототип системы анализа МРТ-снимков головного мозга для задачи бинарной классификации рассеянного склероза (РС).

> **Дисклеймер:** проект создан в учебных и исследовательских целях. Он не является медицинским изделием, не проходил клиническую валидацию и не предназначен для постановки диагноза, лечения или принятия врачебных решений. Любые выводы модели нельзя интерпретировать как медицинское заключение.

---

## О проекте

Цель проекта — отработать полный пайплайн работы с медицинскими изображениями: от чтения DICOM-файлов и предобработки 3D-МРТ-серий до обучения глубокой модели для классификации наличия рассеянного склероза.

Проект моделирует ситуацию, когда МРТ-снимки поступают с разных аппаратов и представлены в виде наборов DICOM-срезов. Основная задача — преобразовать эти данные в пригодный для обучения формат и обучить 3D-свёрточную модель.

## Задача

- Вход: набор DICOM-снимков МРТ головного мозга пациента.
- Выход: бинарная метка — есть ли у пациента рассеянный склероз.
- Особенность: работа не с отдельными 2D-снимками, а с 3D-сериями срезов.

## Данные

### Источник и структура

| Источник данных | Ссылка | Тип данных | Кол-во записей | Тип разметки | Классы разметки | Распределение по классам, % | Дата публикации |
|---|---|---|---|---|---|---|---|
| ГБУЗ «НПКЦ ДиТ ДЗМ», mosmed.ai | https://mosmed.ai/datasets/aie21selftestmri/ | DICOM, МРТ-снимки | 172 | Бинарная: с патологией / без патологии | 1/0 | 50/50 | 22.10.2022 |

Ожидаемая структура каталогов:

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

## Модель

### Архитектура

В основе модели — 3D U-Net-подобная архитектура, адаптированная под задачу классификации 3D-объёмов МРТ.

Основные блоки:

- **DoubleConv** — два последовательных 3D-свёрточных слоя `3 × 3 × 3` с BatchNorm3d и ReLU. Используется как базовый строительный блок.
- **Down** — понижающий блок: `MaxPool3d(2 × 2 × 2)` + `DoubleConv`. Уменьшает пространственное разрешение и увеличивает число каналов.
- **Up** — повышающий блок: `Upsample / ConvTranspose3d` + конкатенация skip-connection с соответствующим уровнем энкодера + `DoubleConv`. Восстанавливает пространственное разрешение.
- **OutConv** — выходной слой `1 × 1 × 1`, приводит число каналов к числу классов.

### Вход и выход

- Вход модели: `[B, 3, 20, 200, 200]`
  - `B` — размер батча;
  - `3` — канала (один 3D-объём дублируется трижды, чтобы соответствовать ожиданиям архитектуры);
  - `20` — число срезов в объёме;
  - `200 × 200` — пространственное разрешение среза.

- Выход модели в текущей версии: `[B, 1, 20, 200, 200]`
  - модель возвращает 3D-объём вероятностей после сигмоиды.

### Функция потерь и оптимизатор

- Loss: `BCELoss` (бинарная кросс-энтропия).
- Оптимизатор: `Adam` с `lr = 0.001`.
- Обучение: до `40` эпох с ранней остановкой при `loss < 0.3` после 10-й эпохи.

### Метрики

На validation и test считаются:

- Accuracy
- Precision
- Recall
- F1-score

## Стек

- Python
- PyTorch
- pydicom
- NumPy
- pandas
- scikit-image
- OpenCV
- scikit-learn
- Matplotlib

## Что не сработало и почему

Честный итог исследования: значимых результатов добиться не удалось. Причина — комбинация неподходящей архитектуры и сложностей датасета.

### Модель

Выбранная 3D U-Net — это **сегментационная** архитектура: она проектировалась под попиксельную разметку и возвращает объём той же размерности, что и вход. Для бинарной классификации (одна метка на пациента) она избыточна и не по назначению:

- декодер и skip-connections не нужны, если ответ — не изображение;
- выход модели `[B, 1, 20, 200, 200]` — это 3D-объём вероятностей, а не метка пациента;
- в обучении есть технические проблемы: несовпадение размеров `outputs` и `labels` в `BCELoss`, нестабильный `squeeze()` без указания оси, дублирование одного 3D-объёма в 3 канала без новой информации, эвристическая гамма-коррекция вместо нормализации.

### Датасет

- **Маленький размер** — типичная проблема медицинских выборок по РС; для 3D-сетей с миллионами параметров этого мало.
- **Разные аппараты** — без harmonization (z-score per-scanner, histogram matching, ComBat) модель учится различать сканеры, а не патологию.
- **Разные протоколы** (T1, T2, FLAIR, с контрастом и без) — при смешивании модель путается.
- **Разметка на пациента, а не на очаг** — концептуальный конфликт с сегментационной архитектурой.

### Как это исправить

**Модель:**
- заменить U-Net на классификационные 3D-сети;
- добавить `AdaptiveAvgPool3d(1)` + `Linear(features, 1)` на выход;
- использовать `BCEWithLogitsLoss` с `pos_weight` для дисбаланса классов;
- убрать дублирование канала — подавать один 3D-объём.

**Данные:**
- нормализовать интенсивности per-volume (z-score внутри маски мозга) вместо гаммы;
- ресемплить к фиксированному `spacing` через `SimpleITK` или MONAI transforms, а не обрезать вслепую;
- добавить harmonization между аппаратами;
- разделять протоколы или явно их кодировать;
- добавить аугментации: random flip, rotation, intensity shift;

### Вывод

Отсутствие результатов — это не провал модели, а следствие неподходящей архитектуры под задачу классификации и сложного, неоднородного датасета. Основные уроки: архитектуру выбирают под тип задачи, а не наоборот; а для медицинских изображений предобработка и harmonization данных значат не меньше, чем сама модель.
