# TorchLeet

TorchLeet is a curated set of PyTorch practice problems, inspired by LeetCode-style challenges, designed to enhance your skills in deep learning and PyTorch.

## Features
- **Diverse Questions**: Covers beginner to advanced PyTorch concepts (e.g., tensors, autograd, CNNs, GANs, and more).
- **Modular Design**: Each question has a unique ID and is self-contained.
- **Guided Learning**: Includes incomplete code blocks (`...` and `#TODO`) for hands-on practice.
- **Solutions Provided**: Answers are available for verification in a separate file.

## Getting Started

### 1. Install Dependencies
Install the required dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Structure
- `questions/`: Contains individual questions (`q<ID>.py`).
- `solutions/`: Corresponding answers (`q<ID>_solution.py`).

### 3. How to Use
- Navigate to questions/ and pick a problem (e.g., q001.py).
- Fill in the missing code blocks `(...)` and address the `#TODO` comments.
- Test your solution and compare it with the corresponding file in `solutions/`.

### 4. Verify
Compare your implementation with solutions/q001_solution.py for correctness.

**Happy Learning! 🚀**

## Question Set

### 🟢Easy
1. [Implement linear regression](https://github.com/Exorust/TorchLeet/blob/main/q1-lin-regression.ipynb) [(Solution)](https://github.com/Exorust/TorchLeet/blob/main/solutions/q1-lin-regression_SOLN.ipynb) 
2. Write a custom Dataset and Dataloader for a CSV file  
3. Write a custom activation function  
4. Write a custom Loss function (Huber Loss)  
5. Zero-out gradients in PyTorch  
6. Run TensorBoard with PyTorch  
7. Save and load your model for later  

### 🟡Medium
8. Implement a Deep Neural Network  
9. Implement an LSTM  
10. Implement a CNN on CIFAR-10  
11. Implement an RNN  
12. Use `torchvision.transforms` to apply data augmentation  
13. Add a benchmark to your PyTorch code  
14. Train an autoencoder for anomaly detection  

### 🔴Hard
15. Write a custom Autograd function  
16. Write a Neural Style Transfer  
17. Write a Transformer  
18. Write a GAN  
19. Write Sequence-to-Sequence with Attention  
20. Quantize your language model
21. Enable distributed training in pytorch (DistributedDataParallel)
22. Work with Sparse Tensors
23. Implement Mixed Precision Training using torch.cuda.amp
24. Add GradCam/SHAP to explain the model.


# Contribution
Feel free to contribute by adding new questions or improving existing ones. Ensure that new problems are well-documented and follow the project structure.

