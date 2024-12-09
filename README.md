# MedLMs
 Medical grade LLM finetuning

![Project Banner](image.png)

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
```

### 🚀 Running the Code
To train and evaluate models, use:
```bash
python train_and_evaluate.py --config configs/llama_config.json
```

To test inference time:
```bash
python inference_benchmark.py --model_path models/llama_3b.pt
```

---

## Results

### Tables
| Model          | BLEU Score | ROUGE Score | Accuracy (%) | Avg. Inference Time (s) |
|-----------------|------------|-------------|--------------|--------------------------|
| **Llama 3-8B**  | 0.59       | 0.58        | 78.5         | 9.5                      |
| **Gemma 1.1-7B**| 0.54       | 0.52        | 74.0         | 10.8                     |
| **DistilGPT2**  | 0.31       | 0.29        | 50.2         | 16.5                     |

### Figures
1. **Model Performance Over Epochs**:
   - Demonstrates learning curves for BLEU, ROUGE, and Accuracy metrics.
   - Performance saturates after **epoch 15**.

2. **Inference Time Comparison**:
   - Highlights the efficiency of the **RTX8000** for medical AI workloads.

---

## Project Structure
```
.
├── configs/
│   ├── llama_config.json
│   ├── gemma_config.json
├── data/
│   ├── processed/
├── images/
│   ├── model_performance_metrics.png
│   ├── inference_time_configs.png
├── models/
├── notebooks/
│   ├── performance_analysis.ipynb
├── results/
├── train_and_evaluate.py
├── inference_benchmark.py
├── README.md
```

---

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Special thanks to:
- **Hugging Face** for providing pre-trained models.
- **NVIDIA** for hardware configurations used in experiments.

For detailed results and models, check our [Hugging Face repository](https://huggingface.co/Apurva3509).
