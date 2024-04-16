# MAKE MORE

# Bigram Level Character Language Model

This project implements a bigram level character language model using PyTorch. The model predicts the next character in a sequence based on the preceding character, aiming to learn the underlying patterns and structure of the text data.

## Project Overview

- **Objective**: Train a bigram language model and generate text sequences based on learned probabilities.
- **Dataset**: Utilize a corpus of text data for training and evaluation.
- **Techniques**: Implement counting-based and neural network-based approaches for modeling bigram probabilities.
- **Tools**: Python, PyTorch, NumPy.

## Project Structure

1. **Data Preparation**:
   - Load and preprocess the text data.
   - Convert characters to indices for model input.
2. **Counting Approach**:
   - Calculate bigram probabilities using counting techniques.
   - Visualize bigram probabilities using a heatmap.
3. **Neural Network Approach**:
   - Design a neural network to learn bigram probabilities.
   - Train the neural network using gradient descent.
4. **Text Generation**:
   - Generate text sequences using the trained model.
   - Compare generated text with the original data for coherence and quality.
5. **Evaluation**:
   - Measure model performance using negative log likelihood (NLL) as the loss metric.
   - Fine-tune model hyperparameters for better performance.

## Usage

1. **Data Preparation**:
   - Load your text data and preprocess it to remove noise and special characters.
   - Convert characters to indices for model input.
2. **Counting Approach**:
   - Use counting techniques to calculate bigram probabilities.
   - Visualize the probabilities to understand the language structure.
3. **Neural Network Training**:
   - Define and train a neural network to learn bigram probabilities.
   - Optimize the network using gradient descent and adjust learning rates as needed.
4. **Text Generation**:
   - Generate text sequences using the trained model.
   - Experiment with different starting sequences to observe text diversity.
5. **Evaluation and Fine-Tuning**:
   - Evaluate the model's performance using negative log likelihood (NLL).
   - Fine-tune hyperparameters such as regularization strength for better results.

## Results

- The project demonstrates the effectiveness of both counting-based and neural network-based approaches in modeling bigram probabilities.
- Text generation results showcase the model's ability to generate coherent and contextually relevant sequences.
- Evaluation metrics such as negative log likelihood (NLL) provide insights into model performance and guide hyperparameter tuning.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (for visualization)

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
