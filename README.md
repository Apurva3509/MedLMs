# MedLMs
 Medical grade LLM finetuning

![Project Banner](https://via.placeholder.com/1200x400.png?text=Medical+Model+Performance+Analysis)

This repository contains the code, results, and insights from analyzing the performance of advanced language models for medical domain tasks. Models such as **Llama 3-8B**, **Gemma 1.1-7B**, and **DistilGPT2** are fine-tuned and evaluated based on BLEU, ROUGE, and accuracy metrics across multiple configurations.

---

## Key Highlights

### 🔑 Key Features:
- Comprehensive **BLEU**, **ROUGE**, and **Accuracy** comparison for models.
- Analysis of **hardware configurations** (RTX8000, A100, V100) for inference efficiency.
- Evaluation of **model architecture** impact on task-specific performance.

### 📊 Key Findings:
- Larger models like **Llama 3-8B** outperform smaller models due to better architecture and pre-training.
- **RTX8000** shows optimal inference performance, while **A100** exhibits unexpected delays.
- Rapid learning during early epochs (5–7) with saturation at epoch 15.

---

## Performance Metrics

### Model Performance Metrics Over Epochs
![Model Performance](images/model_performance_metrics.png)

- **Figure 1**: BLEU, ROUGE, and Accuracy scores across training epochs. Performance saturates after epoch 15 for most models.

### Inference Time Across Configurations
![Inference Time](images/inference_time_configs.png)

- **Figure 2**: Average inference times for models under various configurations (RTX8000, A100, V100). RTX8000 is the most efficient.

---

## Setup and Usage

### 🛠 Requirements
- Python >= 3.8
- PyTorch >= 1.9.0
- Hugging Face Transformers Library
- CUDA-enabled GPUs

### 📦 Installation
Clone this repository and install the required packages:
```bash
git clone https://github.com/username/medical-model-performance.git
cd medical-model-performance
pip install -r requirements.txt
