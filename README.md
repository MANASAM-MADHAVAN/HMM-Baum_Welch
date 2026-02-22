# Hidden Markov Model (HMM) – Baum Welch Algorithm

## Student Details

**Name:** Manasa M  
**Register Number:** TCR24CS044  
**Course:** Pattern Recognition  
**Semester:** S4 CSE  

---

## Project Description

This project implements the **Hidden Markov Model (HMM)** using the **Baum–Welch Algorithm**, which is an Expectation-Maximization (EM) technique used to estimate unknown parameters of an HMM.

The algorithm learns the model parameters that maximize the likelihood of a given observed sequence.

---

## Inputs

- Observed state sequence  
- Number of hidden states  
- Number of training iterations  

---

## Outputs

- Transition Matrix (A)  
- Emission Matrix (B)  
- Initial State Distribution (π)  
- Final Probability P(O | λ)  
- Likelihood values over iterations  
- Likelihood vs Iterations graph  
- State transition diagram  

---

## Visualization

1. **Likelihood vs Iterations Graph**
   - Shows how the probability P(O | λ) increases during each iteration of Baum–Welch training.

2. **State Transition Diagram**
   - Displays hidden states and transition probabilities using a directed graph.

---

## Requirements

- Python 3.x  
- numpy  
- matplotlib  
- networkx  

---

## Installation

Install required packages using:

```bash
pip install numpy matplotlib networkx

---

## How to run

python hmm.py