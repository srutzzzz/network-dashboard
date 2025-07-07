# Network Traffic Analysis Dashboard (UNSW-NB15)

This project is an interactive web dashboard based on Dash. It allows users to explore, analyze, and model network traffic data with the UNSW-NB15 dataset. The project combines exploratory data analysis (EDA), unsupervised anomaly detection, and supervised machine learning (Random Forest) to classify normal and attack traffic.

---

## Features

- Feature Importance insights   
- Clean interactive dashboard with Plotly Dash  
- Exploratory plots of traffic patterns  
- Anomaly detection using Isolation Forest  
- Predictive classification with Random Forest  
- Confusion Matrix to evaluate predictions  

---

## Technologies Used

- Python 3
- Dash (by Plotly)
- Plotly
- Pandas
- Scikit-learn
- Dash Bootstrap Components

---

## Data

The dataset used is the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), a modern network intrusion dataset with multiple attack categories.  
(Note: for large CSV files, you may keep only a sample in this repo to avoid storage issues.)

---

## How to Run Locally

1. Clone this repository:

```bash
git clone https://github.com/yourusername/network-dashboard.git
cd network-dashboard
```