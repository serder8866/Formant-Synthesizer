# Discrete-Time Formant Synthesizer

This repository contains a Python implementation of two discrete-time formant synthesizers: a **cascade synthesizer** and a **parallel synthesizer**, designed to model the vocal tract filter and generate synthetic vowel sounds.

## 📌 Overview

The project synthesizes four vowel types (`/i/`, `/a/`, `/u/`, and schwa) using two approaches:

- **Cascade Synthesizer:** Formant resonators are connected in series. Each resonator shapes the output of the previous one.
- **Parallel Synthesizer:** Resonators operate independently with frequency-dependent gain and are summed together with alternating polarity.

## ⚙️ Features

- Implements second-order IIR filters as formant resonators.
- Supports flexible vowel configuration through predefined formant frequencies and bandwidths.
- Outputs audio files for both cascade and parallel synthesis (with 3, 4, or 5 formants).
- Compares magnitude and phase responses across synthesizers.
- Generates amplitude values and plots for analysis.

## 🧪 Technologies

- **Python 3.11.11** (Miniconda-managed environment)
- **Jupyter Notebook** and **Visual Studio Code**
- Libraries used:
  - [`numpy`](https://numpy.org/)
  - [`scipy`](https://scipy.org/)
  - [`soundfile`](https://pysoundfile.readthedocs.io/)
  - [`matplotlib`](https://matplotlib.org/)

## 🔊 Output

The synthesizers generate `.flac` files representing the output of:

- Cascade synthesis (5 formants)
- Parallel synthesis with 3, 4, and 5 formants

The code also prints and stores formant peak amplitude values in `.txt` files.

## 📁 Structure

- `formant_synthesizer.py`: Main script with signal generation, filter implementation, analysis, and plotting.
- Output files:
  - `*.flac`: Audio output from each synthesizer configuration.
  - `*_amplitude_values.txt`: Cascade synthesizer formant peak values.
  - `*_gain_values_3/4/5.txt`: Parallel synthesizer peak values.

## 🚀 Usage

1. Clone the repository.
2. Install dependencies:  
   ```bash
   pip install numpy scipy matplotlib soundfile
