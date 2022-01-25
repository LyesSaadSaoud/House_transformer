# Household Energy Consumption Prediction Using the Stationary Wavelet Transform and Transformers
This repo contains the supported files to reproduce the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9672113) results using the stationary wavelet transform and deep transfomers 

## ABSTRACT
In this paper, we present a new method for forecasting power consumption. Household power consumption prediction is essential to manage and plan energy utilization. This study proposes a new technique using machine learning models based on the stationary wavelet transform (SWT) and transformers to forecast household power consumption in different resolutions. This approach works by leveraging self-attention mechanisms to learn complex patterns and dynamics from household power consumption data. The SWT and its inverse are used to decompose and reconstruct the actual and the forecasted household power consumption data, respectively, and deep transformers are used to forecast the SWT subbands. Experimental findings show that our hybrid approach achieves superior prediction performance compared to the existing power consumption prediction methods.


 ![Transformer_sans_shift drawio (6)](https://user-images.githubusercontent.com/78357759/150910489-20ede67c-cd30-4977-b552-42a86235ff81.png)

The proposed deep transformer SWT model for the household power consumption forecasting.

## Getting started
1. Install Python 3.7, Tensorflow 2.4, and Keras 2.4
2. Download all files and put them in the same folder. 
3. cd to path. 
4. Run the file [Transformers_Houses1to5_5min.py](https://github.com/LyesSaadSaoud/Wind_forecast/blob/main/main.m) 

Please cite it as: L. Saad Saoud, H. Al-Marzouqi and R. Hussein, "Household Energy Consumption Prediction Using the Stationary Wavelet Transform and Transformers," in IEEE Access, vol. 10, pp. 5171-5183, 2022, doi: 10.1109/ACCESS.2022.3140818.
