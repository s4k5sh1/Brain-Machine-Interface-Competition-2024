# **Neural Decoder for Prosthetic Device**
Submitted on 14 May 2024
Group Name : Firing Squad

## **Description**

This project involves designing a neural decoder to drive a hypothetical prosthetic device, a challenging task in brain-machine interfacing. The goal is to create an algorithm that estimates the precise trajectory of a monkey's hand as it reaches for a target, using spike trains recorded from the monkey's brain. This task simulates a real-world scenario in which neural signals are used to control prosthetic limbs.

## **Task Details**

- **Data**: Spike trains recorded from a monkey's brain during an arm movement task.
- **Objective**: Estimate the X and Y positions of the monkey's hand at each moment in time. Although the training data includes the Z position, the task focuses only on the X and Y positions.
- **Causality**: The decoder must be causal; it cannot use future information to estimate the current hand position.

## **Team Contributions**

- **Sorrana**: Implemented Naive Bayes and K-Nearest Neighbors (KNN) algorithms.
- **Christina**: Developed Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) methods.
- **Sakshi**: Worked on Linear Regression and Principal Component Regression (PCR).
- **Fiona**: Focused on Feature Space selection and Support Vector Machines (SVM).

## **Results and Achievements**

Our neural decoder model achieved significant results:
- **Baseline Model (PCA-LDA-KNN)**: Achieved an accuracy of 99.42%.
- **Final PCR Model**: Used to decode the hand x and y positions, resulting in an RMSE of 7.3017.
- **Model Robustness**: A k-fold cross-validation yielded a mean RMSE of 6.8035.

These results led our team to achieve the **lowest RMSE** in the competition, earning us the title of **Best Group Project**.

---
