# MedLMs
 Medical grade LLM finetuning

![Project Banner](image2.png)

This repository contains the code, results, and insights from analyzing the performance of advanced language models for medical domain tasks. Models such as **Llama 3-8B**, **Gemma 1.1-7B**, and **DistilGPT2** are fine-tuned and evaluated based on BLEU, ROUGE, and accuracy metrics across multiple configurations.

---

## Key Highlights

### Key Features:
- Comprehensive **BLEU**, **ROUGE**, and **Accuracy** comparison for models.
- Analysis of **hardware configurations** (RTX8000, A100, V100) for inference efficiency.
- Evaluation of **model architecture** impact on task-specific performance.

### Key Findings:
- Larger models like **Llama 3-8B** outperform smaller models due to better architecture and pre-training.
- **RTX8000** shows optimal inference performance, while **A100** exhibits unexpected delays.
- Rapid learning during early epochs (5–7) with saturation at epoch 15.

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
git clone https://github.com/Apurva3509/medical-model-performance.git
cd Files
```

### 🚀 Running the Code
To train and evaluate models, deploy them from hugging face and play with the parameters to see the responses for differnet inputs.

---


### Tables
----------------------------------------------------------------------------------------
| Model           | BLEU Score | ROUGE Score | Accuracy (%) | Avg. Inference Time (s)  |
|-----------------|------------|-------------|--------------|--------------------------|
| **Llama 3-8B**  | 0.59       | 0.58        | 78.5         | 9.5                      |
| **Gemma 1.1-7B**| 0.54       | 0.52        | 74.0         | 10.8                     |
| **DistilGPT2**  | 0.31       | 0.29        | 50.2         | 16.5                     |
----------------------------------------------------------------------------------------

## Project Structure
```
MedLMs/
├── Files/
│   ├── .DS_Store
│   ├── DistilGPT_finetuning.ipynb
│   ├── DistilGPT_finetuning_v1.2.ipynb
│   ├── Distilgpt2_vs_Llama3.ipynb
│   ├── Evaluating_finetuned-distill-vs-gemma.ipynb
│   ├── Evaluating_finetuned_LLMs-distilgpt2.ipynb
│   ├── Evaluating_finetuned_LLMs-v1.0.ipynb
│   ├── Llama_2_7b_MedLM.ipynb
│   ├── Mistral_7b_MedLM.ipynb
│   ├── README.md
│   ├── gemma_1.1_7b_it_medical.ipynb
│   ├── llama_3_8b_instruct_medical-new.ipynb
│   ├── medical_models_evaluation.csv
│   └── results-images/
│       ├── .DS_Store
│       ├── Accuracy vs Number of Epochs.png
│       ├── BLEU Scores vs Number of Epochs.png
│       ├── ROUGE Scores vs Number of Epochs.png
│       ├── W&B Chart 11_30_2024, 10_56_31 PM.png
│       ├── W&B Chart 11_30_2024, 10_56_31 PM.svg
│       ├── gpu_inference_comparison.html
│       ├── newplot-v1.png
│       └── newplot-v2.png
├── Presentation/
│   ├── .DS_Store
│   ├── EECS E6694 Final Presentation.pdf
│   ├── EECS E6694 Final Presentation.pptx
├── Report/
│   ├── E6694 GenAI Report.zip
│   ├── E6694_GenAI_Report.pdf
├── artifacts/
│   ├── all_models.png
│   ├── gemma_loss.csv
│   ├── gemma_loss.png
│   ├── llama2_loss.csv
│   ├── llama2_loss.png
│   ├── llama3_loss.csv
│   ├── llama3_loss.png
│   ├── mistral_loss.csv
│   ├── mistral_loss.png
│   ├── trainer_stats_gemma.json
│   ├── trainer_stats_llama2.json
│   ├── trainer_stats_llama3.json
│   ├── trainer_stats_mistral.json
├── Data/
│   ├── instruct_datasets.py
│   ├── medical_meadow_wikidoc.csv
│   ├── medquad.csv
├── .DS_Store            
├── .gitattributes      
├── .gitignore           
├── LICENSE             
├── README.md            
├── image.png            
├── image2.png           
├── requirements.txt
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

For detailed results and models, check our Hugging Face repository:
[Apurva](https://huggingface.co/Apurva3509)
[Abhilash](https://huggingface.co/abhilash2599)
