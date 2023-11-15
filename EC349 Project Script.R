#EC349 Project Script

#Clear
cat("\014")  
rm(list=ls())

#Load Libraries
library(glmnet)
library(ggplot2)
library(tidyverse)

#Loading the small versions of data
load(file='/Users/danieldzhiris/Downloads/EC349 Project/Assignment/Small Datasets/yelp_review_small.Rda')
load(file='/Users/danieldzhiris/Downloads/EC349 Project/Assignment/Small Datasets/yelp_user_small.Rda')
