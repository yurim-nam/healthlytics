# Project 1: Schizophrenia PANSS Analysis

This project is an exploratory data analysis of different psychological test scores (mainly being PANSS) from multiple schizophrenia clinical trials, where patients were treated by different investigators under two treatment conditions (risperidone (atypical) vs. haloperidol (typical - our control)). 

The overall aim has been to explore patterns in symptom change across patients and investigators and to set up the dataset in a way that reflects the true patient–treatment effect.

## Main Objectives 
---
1. **Clustering Symptom Profiles**
> “Can we identify distinct subtypes of schizophrenia based on PANSS symptom scores?”
2. **Longitudinal Data comparison**
> “Can we use our cluster datasets and compare with our longitudinal data?”
3. **Control vs Risperidone - Treatment effectiveness**
> “Can we see a statistically significant difference in various symptom scores comparing our control vs our treatment of interest?”

## Methods
---
- K-means & PCA (`kmeans()`, `hclust()`, `dplyr`, `factoextra`,`prcomp()` )
- PCA for dimensionality reduction
- Bar Plots & Silhouette Plots

## [Enjoy the code](https://yurim-nam.github.io/healthlytics/docs/PANSS_Analysis/PANSS_Analysis_book.html)

## Data Source 
---
(from the same Surrogate package)

[PANSS dataset](https://r-packages.io/datasets/PANSS)

[Schizo dataset](https://r-packages.io/datasets/Schizo)
