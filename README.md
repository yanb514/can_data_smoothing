# can_data_smoothing

This repository provides a tool for smoothing CAN bus data, incluing position, velocity and acceleration, while preserving internal consistency.
For details of problem formulation please refer to the PDF (https://github.com/yanb514/can_data_smoothing/blob/main/can_bus_smoothing_doc.pdf).

### Requirement
- CVXOPT: https://cvxopt.org/userguide/index.html

### Tuning parameters
- lam1: controls the penalty for velocity error. Make this higher if velocity data is more trustworthy
- lam2: controls the penalty for acceleration error.  Make this higher if acceleration data is more trustworthy
- lam3: controls jerk regularization.  Make this higher for smoother signals

Note that there is a trade-off amongst data fitting and smoothing.

### Results with different parameter choices
lam1 = 1, lam2 = 10, lam3 = 10
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/1_10_10.png)

lam1 = 1, lam2 = 10, lam3 = 100
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/1_10_100.png)

lam1 = 1, lam2 = 10, lam3 = 1000
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/1_10_1000.png)

lam1 = 1, lam2 = 10, lam3 = 10000
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/1_10_10000.png)

### This version handles
- complete time series data, no missing data
- uniform timestamps (e.g., 10Hz)
- forward Euler numerical discretization method
- measurement noise, no sparse outliers (otherwise, a L1 regularization is necessary)

It is necessary to standardize the data into the above format before applying this smoother. Additional features can be made upon request.
