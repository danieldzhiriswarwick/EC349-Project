#EC349 Project Script
#Clear
cat("\014")  
rm(list=ls())

#Load the libraries
library(dplyr)
library(ggplot2)
library(tm)
library(caret)
library(tidytext)
library(broom)
library(glmnet)
library(tidyverse)
library(tree)
library(rpart)
library(rpart.plot)
library(knitr)
library(stringr)
library(textdata)
library(wordcloud)
library(randomForest)
library(nnet)
library(ipred)
library(jsonlite)
library(MASS)

#Set Directory as appropriate
setwd("/Users/danieldzhiris/Downloads/EC349 Project/Assignment")

#Load Different Data
business_data <- stream_in(file("yelp_academic_dataset_business.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
checkin_data  <- stream_in(file("yelp_academic_dataset_checkin.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
tip_data  <- stream_in(file("yelp_academic_dataset_tip.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
user_data  <- stream_in(file("yelp_academic_dataset_user.json")) #note that stream_in reads the json lines (as the files are json lines, not json)

load(file='/Users/danieldzhiris/Downloads/EC349 Project/Assignment/Small Datasets/yelp_review_small.Rda')

#Exploration of the Data
head(review_data_small, 5)
glimpse(review_data_small)
dim(review_data_small)
glimpse(business_data)
glimpse(user_data)
glimpse(tip_data)
glimpse(checkin_data)

# Number of rows (observations)
nrow(review_data_small)

# Number of observations by stars rating
stars_counts <- table(review_data_small$stars)
print(stars_counts)

#Plot frequency distribution of star ratings
#Sample is skewed towards 4 and especially 5 star reviews
ggplot(review_data_small, aes(x = stars)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(x = "Rating",
       y = "Number of Reviews (million)") +
  scale_y_continuous(labels = scales::number_format(scale = 1e-6, suffix = "M"))

# Calculate the percentage of reviews for each star rating
percentage_data <- review_data_small %>%
  group_by(stars) %>%
  summarise(percentage = n() / nrow(review_data_small) * 100)

# Create a pie chart
ggplot(percentage_data, aes(x = "", y = percentage, fill = as.factor(stars))) +
  geom_bar(stat = "identity", width = 1) +
  geom_text(aes(label = sprintf("%.1f%%", percentage)),
            position = position_stack(vjust = 0.5), size = 3) +
  coord_polar("y") +
  labs(title = "Percentage of Reviews by Star Rating",
       fill = "Rating") +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5))

#Set seed for reproducibility
set.seed(123)
#Selecting random 30,000 observations to make the analysis computationally feasible
rev_small_index <- sample(nrow(review_data_small), 30000)
rev_data <- review_data_small[rev_small_index, ]

#Merge with business rating from business_data
colnames(business_data)[colnames(business_data) == "stars"] <- "business_rating"
rev_data <- rev_data %>%
  left_join(dplyr::select(business_data, business_id, business_rating), by = 'business_id')

#Box-and-whisker plot for the distribution of stars according to business rating
#It is evident that there is correlation between the star rating and the business rating
ggplot(rev_data, aes(x = factor(stars), y = business_rating)) +
  geom_boxplot() +
  labs(x = "Stars", y = "Business Rating") +
  ggtitle("Stars vs Business Rating")

#Merge with the average ratings given by a user from user_data_small
rev_data <- rev_data %>%
  left_join(dplyr::select(user_data, user_id, average_stars), by = 'user_id')

#Box-and-whisker plot for the distribution of the average of ratings given by the user
ggplot(rev_data, aes(x = factor(stars), y = average_stars)) +
  geom_boxplot() +
  labs(x = "Stars", y = "Average of Ratings Given by a User") +
  ggtitle("Stars vs Average of Ratings Given by the User")

#SENTIMENT ANALYSIS
#Keeping only the text variable to make sentiment analysis computationally efficient
text_rev_data <- rev_data %>%
  dplyr::select(review_id, text)

#Excluding the following words from the stop_words lexicon.
words_to_remove <- c("unfortunately", "great", "good", "greatest", "matter")
custom_stop_words <- stop_words %>%
  filter(!word %in% words_to_remove)

#Count all words used in reviews 
review_words <- text_rev_data %>%
  unnest_tokens(word, text) %>%
  anti_join(custom_stop_words) %>%
  count(review_id, word, sort = TRUE)

#Count the total number of "sentiment" tokens in each review
total_words <- review_words %>%
  group_by(review_id) %>%
  summarize(total_tokens = sum(n))

review_words <- left_join(review_words, total_words)

#Load the AFINN lexicon for sentiment analysis
afinn_lexicon <- get_sentiments("afinn")

#Bigrams
#Evaluation of how often sentiment-associated words are preceded by “not” or other negating words
rev_bigrams <- text_rev_data %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  filter(!is.na(bigram))

#Separating bigrams by word1 and word2
bigrams_separated <- rev_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

#Most often used combinations of 2 words excluding stop words
bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% custom_stop_words$word) %>%
  filter(!word2 %in% custom_stop_words$word)

bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)
head(bigram_counts, 10)

#Bigrams that start with "not"
not_bigrams_separated <- bigrams_separated %>%
  filter(word1 == "not") %>%
  filter(!word2 %in% custom_stop_words$word) %>%
  count(word1, word2, sort = TRUE)
head(not_bigrams_separated)

#Contribution of "not" bigrams to total sentiment of all reviews
not_words <- bigrams_separated %>%
  filter(word1 == "not") %>%
  inner_join(afinn_lexicon, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)
head(not_words, 10)

not_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(n * value, word2, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  labs(x = "Sentiment value * number of occurrences",
       y = "Words preceded by \"not\"")

#All identified negation words
negation_words <- c("not", "no", "never", "without", "isn't", "aren't", "doesn't", "don't",
                    "didn't", "hasn't", "haven't")

negated_words_count <- bigrams_separated %>%
  filter(word1 %in% negation_words) %>%
  filter(!word2 %in% custom_stop_words$word) %>%
  inner_join(afinn_lexicon, by = c(word2 = "word")) %>%
  count(word1, word2, value, sort = TRUE)
head(negated_words_count, 10)

negated_words_count %>%
  mutate(contribution = n * value) %>%
  group_by(word2) %>%
  summarize(total_contribution = sum(contribution)) %>%
  arrange(desc(abs(total_contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, total_contribution)) %>%
  ggplot(aes(total_contribution, word2, fill = total_contribution > 0)) +
  geom_col(show.legend = FALSE) +
  labs(x = "Sentiment value * number of occurrences",
       y = "Words preceded by negation words")

#AFINN Values to be reversed due to the presence of negation words
negated_words <- bigrams_separated %>%
  filter(word1 %in% negation_words) %>%
  filter(!word2 %in% custom_stop_words$word) %>%
  inner_join(afinn_lexicon, by = c(word2 = "word"))

#Summarising total AFINN Values to be reversed due to the presence of negation words by review
negated_sent <- negated_words %>%
  group_by(review_id) %>%
  summarize(value = sum(value, na.rm = TRUE))

#TF_IDF (not used)
#review_tf_idf <- review_words %>%
#  bind_tf_idf(word, review_id, n)
#review_tf_idf <- review_tf_idf %>%
#  inner_join(afinn_lexicon, by = c("word" = "word"))
#review_sent <- review_tf_idf %>%
#  group_by(review_id) %>%
#  summarize(review_sentiment = sum(tf_idf * value, na.rm = TRUE))
#review_sent <- merge(x = review_sent, y = rev_data[, c("review_id", "stars")], 
#                    by = "review_id", all.x = TRUE)
#

#Total AFINN sentiment for each review (not accounting for negation words)
afinn_review_sent <- review_words %>%
  inner_join(afinn_lexicon, by = c("word" = "word")) %>%
  group_by(review_id) %>%
  summarize(afinn_sentiment = sum(value, na.rm = TRUE))

#Merging review sentiment scores for each review with respective star ratings
afinn_review_sent <- merge(x = afinn_review_sent, y = rev_data[, c("review_id", "stars")], 
                           by = "review_id", all.x = TRUE)

#Adding a column for the total number of words in each review
afinn_review_sent <- left_join(afinn_review_sent, review_words %>% 
                                 dplyr::select(review_id, total_tokens), by = "review_id") %>%
  distinct(review_id, .keep_all = TRUE)

#Adding a column with reversed sentiment of words that are preceded by negation words
afinn_review_sent <- afinn_review_sent %>%
  left_join(negated_sent %>% dplyr::select(review_id, value), by = "review_id") %>%
  rename(negated_value = value) %>%
  mutate(sentiment_adjusted = ifelse(is.na(negated_value), afinn_sentiment, 
                                     afinn_sentiment - (2 * negated_value)))

#Normalising sentiment score for the total number of words in a review
afinn_review_sent <- afinn_review_sent %>%
  group_by(review_id) %>%
  mutate(afinn_sentiment_norm = sum(sentiment_adjusted / total_tokens, na.rm = TRUE))

#Merge back the business_rating and average_stars variables 
afinn_review_sent <- afinn_review_sent %>%
  left_join(dplyr::select(rev_data, review_id, business_rating, average_stars), by = 'review_id')

#Exploring the relationship between the length of a review and the stars rating
ggplot(afinn_review_sent, aes(x = total_tokens, y = stars)) +
  geom_point(color = "blue", size = 3) +
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, color = "red", linetype = "solid") +
  labs(title = "Scatter Plot of Total vs. Stars",
       x = "Length of Review", y = "Stars")

ggplot(afinn_review_sent, aes(x = total_tokens, y = stars)) +
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, color = "red", linetype = "solid") +
  labs(title = "Regression Line of Lenth of Review vs. Stars",
       x = "Length of Review", y = "Stars")

#Checking for multicollinearity among the chosen independent variables
independent_var <- afinn_review_sent[, c("afinn_sentiment_norm", "business_rating",
                                         "average_stars", "total_tokens")]
correlation_matrix <- cor(independent_var)
print(correlation_matrix)

#"Stars" as a factor variable
afinn_review_sent$stars <- as.factor(afinn_review_sent$stars)

#Set seed for reproducibility
set.seed(1)
#Split data so that the test data set has 10,000 random observations
test_indices <- sample(1:nrow(afinn_review_sent), 10000, replace = FALSE)
test_data <- afinn_review_sent[test_indices, ]
# Use the remaining observations for the training set
train_data <- afinn_review_sent[-test_indices, ]

# Decision Tree
rpart_tree <- rpart(stars ~ afinn_sentiment_norm+business_rating+average_stars, data = train_data)
rpart.plot(rpart_tree)
rpart_predictions <- predict(rpart_tree, newdata = test_data, type = "class")
rpart_confusion_matrix <- table(rpart_predictions, test_data$stars)
print(rpart_confusion_matrix)
rpart_accuracy <- sum(diag(rpart_confusion_matrix)) / sum(rpart_confusion_matrix)
print(paste("Accuracy:", round(rpart_accuracy, 2)))

# Bagging
bag <- bagging(stars ~ afinn_sentiment_norm+business_rating+average_stars, 
               data = train_data, nbagg = 50)
bag_predictions <- predict(bag, newdata = test_data, type = "class")
bag_confusion_matrix <- table(bag_predictions, test_data$stars)
print(bag_confusion_matrix)
bag_accuracy <- sum(diag(bag_confusion_matrix)) / sum(bag_confusion_matrix)
print(paste("Accuracy:", round(bag_accuracy, 2)))

# Random Forest
set.seed(2)
rf_tree <- randomForest(stars ~ afinn_sentiment_norm+business_rating+average_stars+total_tokens,
                        data = train_data, ntree = 100, mtry = 1, importance = TRUE)
rf_predictions <- predict(rf_tree, newdata = test_data, type = "class")
rf_confusion_matrix <- table(rf_predictions, test_data$stars)
print(rf_confusion_matrix)
rf_accuracy <- sum(diag(rf_confusion_matrix)) / sum(rf_confusion_matrix)
print(paste("Accuracy:", round(rf_accuracy, 2)))
mean(rf_tree[["err.rate"]])
importance(rf_tree)
varImpPlot(rf_tree, main = "Variable Importance Plot")
plot(rf_tree)
# Null rate
null_rate <- max(table(test_data$stars)) / length(test_data$stars)
print(paste("Null Rate:", round(null_rate, 2)))

#Combining the test data with the predicted star ratings to examine observations with incorrect predictions
test_data <- cbind(test_data, rf_predictions)

# Ordered probit
# Fit ordered probit model on train_data
ordered_probit <- polr(stars ~ afinn_sentiment_norm + business_rating + average_stars + total_tokens,
                       data = train_data, method = "probit")
# Predict stars on test_data
probit_predictions <- predict(ordered_probit, newdata = test_data, type = "class")
# View the predicted values
probit_confusion_matrix <- table(probit_predictions, test_data$stars)
print(probit_confusion_matrix)
probit_accuracy <- sum(diag(probit_confusion_matrix)) / sum(probit_confusion_matrix)
print(paste("Accuracy:", round(probit_accuracy, 2)))

# Comparison of OOB Error for tree-based methods
# Set your values for m (number of predictors to consider at each split)
m_values <- c(4, 2, 1)  # Updated values
# Set the initial and final number of trees, as well as the increment
initial_trees <- 1
final_trees <- 101
tree_increment <- 10
# Create an empty data frame to store results
results_df <- data.frame()
# Loop through different values of m
for (m in m_values) {
  # Loop through different numbers of trees
  for (ntree in seq(initial_trees, final_trees, by = tree_increment)) {
    # Create and train a Random Forest model
    rf_model <- randomForest(factor(stars) ~ afinn_sentiment_norm+business_rating+average_stars+total_tokens, 
                             data = train_data, 
                             mtry = m, 
                             ntree = ntree,
                             importance = TRUE)
    
    # Extract test classification error
    error_rate <- mean(rf_model[["err.rate"]][, "OOB"])
    
    # Create a data frame with results
    results_df <- rbind(results_df, data.frame(m = rep(m, 1), 
                                               ntree = ntree, 
                                               test_error = error_rate))
  }
}

# Plot the results
ggplot(results_df, aes(x = ntree, y = test_error, color = factor(m))) +
  geom_line() +
  labs(x = "Number of Trees",
       y = "Out-of-Bag (OOB) Error") +
  scale_color_manual(name = "",
                     values = c("4" = "blue", "2" = "green", "1" = "red"),
                     labels = c("4" = "m = p = 4", "2" = "m = 2", "1" = "m = 1")) +
  theme(legend.position = c(0.85, 0.9),
        panel.grid = element_blank(),
        panel.background = element_blank())