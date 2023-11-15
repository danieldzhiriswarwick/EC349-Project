#EC349 Project Script

#Clear
cat("\014")  
rm(list=ls())

#Load Libraries
library(glmnet)
library(ggplot2)
library(tidyverse)
library(caret)

#Loading the small versions of data
load(file='/Users/danieldzhiris/Downloads/EC349 Project/Assignment/Small Datasets/yelp_review_small.Rda')
load(file='/Users/danieldzhiris/Downloads/EC349 Project/Assignment/Small Datasets/yelp_user_small.Rda')

# Set seed for reproducibility
set.seed(1)

# Select 10,000 random observations for the training data set
train_obs <- sample(nrow(review_data_small), 10000)
train_data <- review_data_small[train_obs, ]

# Select all other observations for the testing data set
test_data <- review_data_small[-train_obs, ]
