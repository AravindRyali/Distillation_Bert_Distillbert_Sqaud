# **Knowledge Distillation for Question-Answering (QA) Models**

This repository contains an implementation of knowledge distillation for Question-Answering (QA) models, using a teacher model (BERT) and a student model (DistilBERT). The goal is to distill knowledge from the larger BERT model into the smaller and more efficient DistilBERT model for use in resource-constrained environments.

## **Table of Contents**
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## **Introduction**
Knowledge distillation allows a smaller student model to learn from the predictions (logits) of a larger teacher model. In this project:
- **Teacher model**: BERT-large (fine-tuned on the SQuAD dataset).
- **Student model**: DistilBERT (not fine-tuned initially, then trained via distillation).

The distillation process helps in reducing the size of the model while retaining much of the performance of the teacher model.

## **Models Used**
- **Teacher Model**: `bert-large-uncased-whole-word-masking-finetuned-squad` (BERT-large, pre-trained and fine-tuned on the SQuAD dataset).
- **Student Model**: `distilbert-base-uncased` (DistilBERT-base, pre-trained but not fine-tuned).

## **Requirements**
- Python 3.8+
- PyTorch
- Transformers library by HuggingFace
- Datasets library by HuggingFace
- tqdm (for progress bars)

## **Installation**

1. Clone this repository:

```bash
git clone https://github.com/yourusername/qa-model-distillation.git
cd qa-model-distillation
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## **Usage**

### **Running the Distillation Process**
To run the knowledge distillation script, make sure your teacher and student models are defined. You can use the pre-trained teacher model and initialize the student model as shown in the script.

1. Prepare your dataset (e.g., SQuAD):
   ```bash
   from datasets import load_dataset
   dataset = load_dataset("squad")
   ```

2. Set up the distillation training loop:
   - Teacher model will predict logits.
   - Student model will learn via the distillation loss function using KL-divergence.

### **Saving and Loading Models**
- After training, the student model is saved using:
   ```python
   student_model.save_pretrained("path/to/save/student_model")
   student_tokenizer.save_pretrained("path/to/save/student_model")
   ```

- To load models from the saved directory:
   ```python
   from transformers import AutoModelForQuestionAnswering, AutoTokenizer
   student_model = AutoModelForQuestionAnswering.from_pretrained("path/to/save/student_model")
   student_tokenizer = AutoTokenizer.from_pretrained("path/to/save/student_model")
   ```

## **Training Process**

1. The training script leverages:
   - Teacher's output (logits) to guide the student model.
   - Distillation loss (using KL divergence) to minimize the difference between student and teacher logits.

2. The dataset used in the demo is a small subset of SQuAD (`train[:1%]` for testing purposes).

3. Example code for distillation is included, with optimization and backpropagation implemented using PyTorch.

## **Evaluation**

After training, you can evaluate the student model by running:

```python
student_output = student_qa_pipeline(question="What is the capital of France?", 
                                     context="France is a country in Europe. Its capital is Paris.")
print("Student Model Output:", student_output)
```

## **Results**

The performance of the student model is expected to be close to the teacher model, while having significantly fewer parameters, making it more efficient for deployment on smaller devices.

## **Contributing**

Feel free to fork this repository, open issues, and submit pull requests. Contributions are welcome!

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.
