# can_data_smoothing

This repository provides a tool for smoothing CAN bus data, incluing position, velocity and acceleration, while preserving internal consistency.
For details of problem formulation please refer to the PDF (https://github.com/yanb514/can_data_smoothing/blob/main/can_bus_smoothing_doc.pdf).

### Requirement
- CVXOPT: https://cvxopt.org/userguide/index.html

### Tuning parameters
There are 3 parameters in smooth_can_data.py
- lam1: controls the penalty for velocity error. Make this higher if velocity data is more trustworthy
- lam2: controls the penalty for acceleration error.  Make this higher if acceleration data is more trustworthy
- lam3: controls jerk regularization.  Make this higher for smoother signals

Note that there is a trade-off between data fitting and smoothing.

### Results with different parameter choices
Ego vehicle
lam1 = 0, lam2 = 0, lam3 = 1
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/ego_0_0_1.png)

Lead vehicle
lam1 = 0, lam2 = 0, lam3 = 1
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/lead_0_0_1.png)

Ego vehicle
lam1 = 1, lam2 = 1, lam3 = 0
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/ego_1_1_0.png)

Lead vehicle
lam1 = 1, lam2 = 0, lam3 = 0
![](https://github.com/yanb514/can_data_smoothing/blob/main/figures/lead_1_0_0.png)

### This version handles
- complete time series data, no missing data
- uniform timestamps (e.g., 10Hz)
- forward Euler numerical discretization method
- measurement noise, no extreme outliers (otherwise, a L1 regularization is necessary)

It is necessary to standardize the data into the above format before applying this smoother. Additional features can be made upon request.
