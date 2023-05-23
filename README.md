# POPULARITY SCORE OF MUSIC

## Costanza Placanica - Giulia Ghislanzoni



## 1. Introduction
The music streaming industry is highly competitive, with companies competing to attract and retain users by providing a wide range of songs and personalized recommendations. To gain a competitive edge, a large music streaming service has tasked us to conduct an exploratory data analysis on a comprehensive dataset of songs, including information on their audio features, popularity, and track, album and artist details. By analyzing this dataset, we aim to uncover insights into the characteristics of popular songs and identify patterns that can inform the development of more accurate recommendation models. Ultimately, our analysis can help the company enhance user engagement and retention by providing tailored music recommendations that align with users' preferences.

Moreover, using different regression techniques, we aim to provide the company with a reliable and efficient predictor for the popularity of the songs, in order to help the business on investing their resources strategically, such as signing artists, promoting certain songs, or securing licensing deals. This can lead to increased sales, streaming numbers, concert ticket sales, merchandise purchases, and other revenue-generating opportunities.

## 2. Methods
![Flowchart.png](Plots%2FFlowchart.png)

### 2.1 Environment
The deliverable is divided into a MAIN.ipynb file, which is a file Jupyter Notebook where we wrote all the code, and this file README.md, where we wrote the report. Furthermore, all the plots displayed in this report will be provided in a separate folder. This will allow them to be easily accessed and displayed within this document as needed.

We utilized Pandas and NumPy libraries for data processing, while Seaborn and Matplotlib libraries were employed for data visualization purposes. The Sklearn library played a significant role in implementing clustering and regression models, as well as evaluating the metrics we utilized.

The code is implemented for Python 3.7 and above. 

### 2.2 Exploratory Data Analysis
Our dataset consists of 21 columns and 114000 observations. We performed data cleaning and removed null values and duplicates, as they represented a negligible proportion compared to the rest of the dataset.

We performed an exploratory data analysis (EDA) on the entire dataset and visualized the data to gain insights into the correlations between different features of songs and their distributions.

Initially, we discovered that some track_ids were repeated in the data. Further investigation revealed that this was due to the presence of multiple track_genre and popularity scores for the same track_id. Additionally, we observed that the album in which a song was present could vary, depending on whether it was the original album or a compilation such as top 50 hits, summer hits, etc. We also noticed that the popularity of a song varied based on the album in which it was present, with greater popularity in the original album. This issue will be our main focus during data pre-processing.

Then, we conducted descriptive statistics and visualizations to explore the distributions of and correlations between various features of the songs.

Firstly, we calculated descriptive statistics and format each statistic as a float with a fixed number of decimal places. As a result, we noticed that song's features variables such has tempo or loudness have a different range of values which are much higher than the others, whose values are mostly between 0 and 1.
![descriptive_statistics.png](Plots%2Fdescriptive_statistics.png)

Secondly, we wanted to explore the correlations among variables. For this purpose, we plotted a heatmap. We found out that most of the variables are not correleted, with the exception of:
- "acousticness" and "energy", which are highly negatively correlated. This is probably because tracks that have a lot of acoustic sounds tend to be more mellow and subdued, which would result in low energy values. Conversely, tracks that have more electronic sounds might be more energetic and upbeat, resulting in high energy values.
- "loudness" and "energy", which are highly positively correlated. Loudness is a measure of the sound pressure level of a track, while energy is a perceptual measure of the intensity and activity of a track. Therefore, the high correlation is reasonable, as both measures are related to the subjective impression of the track being "loud" or "powerful".
![corr.png](Plots%2Fcorr.png)

Then, we decided to explore the distributions of each numerical variable in our dataset. So, we created a histogram for each one, to show the distribution of its values. Thanks to this plot we could notice that most of the variables are not normally distributed and could gain few relevant insights:
- From the "energy" histogram, we notice that songs tend to have higher levels of energy. The higher levels of energy observed in the histogram indicate that the analyzed songs in the dataset generally exhibit a greater sense of intensity, liveliness, and higher perceived activity. These songs are likely to have more energetic elements, such as a strong rhythm, prominent beats, and a more intense sound overall. However, it's important to consider that the energy attribute is a subjective measure and may vary based on individual perception and interpretation.
- Most of the tracks in the dataset have a speechiness value of 0, indicating a lack of spoken words or vocals in the music. Interestingly, instrumentalness, which measures the extent to which a track consists of instrumental sounds without any vocal content, also predominantly has a value of 0. The calculation methods for these metrics are not explicitly clear, as speechiness should represent the amount of text in songs, while instrumentalness should represent the amount of instrumental sound. However, it is worth noting that these two variables are not negatively correlated and exhibit similar distributions, which raises questions about their accurate representation and calculation methods They may not perfectly capture the nuances of speech or instrumental elements in every track.
![features_distributions.png](Plots%2Ffeatures_distributions.png)

Also, we wanted to investigate the range of value of numerical track features. Therefore, we created a boxplot, using the specified features:'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence'. We do not include variables such has tempo or loudness since, from the decriptive analytics, we noticed that they have a different range of values which are much higher than the others, whose values are mostly between 0 and 1. The analysis of the box plots reveals that:
- both 'energy' and 'danceability' exhibit values that are predominantly above the average range.
- 'speechiness' and 'instrumentalness' have low values with several outliers.
- 'acousticness' shows a wide range of values, spanning from very low to over 0.5.
![boxplot_track_features.png](Plots%2Fboxplot_track_features.png)

Then, we wanted to explore the distribution of what will be our target variable in the regression analysis: popularity. So, we created a histogram and kernel density estimate plot for the 'popularity' column, in order to visualize the distribution of its values. We noticed that the distribution of popularity in the dataset suggests that most songs tend to have average popularity ratings, while only a small portion of songs achieve a high level of popularity. The large number of songs with a popularity of 0 indicates that these tracks have not gained significant attention or recognition among listeners. This could be due to various reasons, such as being from lesser-known artists or simply not aligning with current popular trends. Additionally, factors like marketing, promotion, and playlist placements can heavily influence a song's popularity, and songs with limited visibility or accessibility may struggle to gain widespread popularity.
![distribution_of_popularity.png](Plots%2Fdistribution_of_popularity.png)

Additionally, we displayed the top 5 artists per count. As a result we obtained that The Beatles, George Jones, Stevie Wonder, Linkin Park, and Ella Fitzgerald are the most prominent artists based on the frequency of their appearances. They have been featured most frequently in the dataset, highlighting their prominence in the music's context.
![top_5_artists.png](Plots%2Ftop_5_artists.png)

Moreover, we decided to explore which where the top 10 track genres in our dataset. Therefore, we first calculated the average popularity for each track_genre and get the first 10. Then, we plotted a bar plot showing the top 10 genres ranked by average popularity. This analysis reveals that the audience has diverse and eclectic music tastes. Pop-film is the most popular genre, followed by K-Pop, chill music, and sad music.  Indian music and anime music also rank high, showing a preference for traditional and culturally significant music and music associated with Japanese media. What is surprising is the presence of Sertanejo music, a popular genre in Brazil, that also seems to have a significant following among the  audience.
![top_10_genres.png](Plots%2Ftop_10_genres.png)

Also, we wanted to investigate whether the majority of the songs were explicit or not. In order to do so, we displayed in a countplot the percentages of explicit and not explicit songs. More than 90% of the songs in the dataset are not marked as explicit, which indicates that the vast majority of the songs are suitable for all audiences, without any explicit or mature content.
![explicit.png](Plots%2Fexplicit.png)

Finally, we decided to investigate whether the popularity of a song could be influenced by the different key used. We divided the data into two subsets based on the 'mode' column (mode 0 and mode 1), calculated the average popularity for each key within each mode, and created a bar plot comparing the average popularity of keys for minor mode (mode 0) and major mode (mode 1) using subplots. The analysis suggests that there is minimal variation in popularity between them. Usually, songs in a major key tend to have higher popularity scores because major keys tend to sound more upbeat and positive, which can lead to a more widespread appeal and popularity. However, in this case, songs in a minor key, that convey a more melancholic and emotional tone, are successful too.
![keys.png](Plots%2Fkeys.png)

### 2.3 Data Preprocessing
We needed to address the issue of reapeted track ids that, despite being supposed to be unique values, they are clearly repeated throughout the dataset.

Firstly, we performed a count of occurrences for each unique track_id value that appears more than once in the dataset. This reveals that there are many track_id values that are repeated.

To understand the differences among these repeated track_id values, we grouped the dataset by track_id and examine the duplicates in each variable. Upon reviewing the results, we observed that certain track_id values may appear multiple times due to variations in either the popularity score or the assigned track genre. Essentially, each unique track_id represents a specific song with identical values for all variables except for popularity (720 track_id values have different popularity scores) or track_genre (16299 track_id values have different genres). Hence, different rows with the same track_id correspond to different popularity scores or genres assigned to the same song.

We were also interested in identifying cases where both popularity and track_genre differ for a single track_id. Given that there are 15872 track_id values with different genres and popularity, it indicates that the majority of instances involve songs with varying genres and popularity, resulting in multiple repetitions of track IDs.

Secondly, since songs can be represented by different track ids due to their inclusion in different albums, we aimed to explore the variable album_name and determine if popularity is influenced by the album. To accomplish this, we grouped the track_name with the artists variable, as the same song title might be used by different artists but only once by the same singer. Consequently, we created a new dataset containing track_ids, track_names, artists, album_name, and track_genre.

Next, we filtered the df_check dataset to identify duplicate rows based on the 'artists' and 'track_name' columns, retaining all occurrences of duplicates in the resulting df dupl. This enabled us to include only those songs that are repeated but may have differences in other columns.

Subsequently, we selected the duplicate rows in the dupl dataframe based on the combination of 'artists' and 'track_name', and then eliminated any remaining duplicates within these selected rows using the 'track_name', 'artists', and 'album_name' columns. This ensured that we excluded songs that are repeated due to reasons unrelated to the album.

By following these steps, we gained insights into the occurrences of repeated track_id values and their variations in popularity, genre, and album association. We decided to investigate 4 songs. We observed that the popularity of a song is significantly influenced by the album in which it is released, with higher values typically associated with the original album. Based on these results, we made the assumption that the same pattern holds true for the other songs.

However, despite these findings, we decide not to reduce the dataset by considering only one album per song. Instead, we will treat songs from different albums as distinct entities, as we want to avoid the potential loss of a significant amount of valuable data.

Moving forward with our data processing, we pursued a two-fold strategy for dealing with repeated track ids, before proceeding with the clustering and regression analyses:
- Firstly, we created a new column called "genre" that lists the genres in a list format, eliminating the repetition of songs with multiple genres. This approach ensures that each song appears only once with different genres. Subsequently, we removed the original "track_genre" column.
- Secondly, we replaced the popularity values for the same track_id with the maximum popularity value among those entries. This step maintains the uniqueness of popularity values for each song while rectifying any repeated values and substituting them with the most representative value.

With these modifications, we created a new dataframe. We dropped the variables that are no longer needed and exported the modified dataset as a new CSV file. This will allow others to utilize the updated dataset without having to repeat the preceding steps.

Moving on to the new dataset, we rechecked for any duplicated values to ensure data integrity and we verified the uniqueness of track_id values. Remarkably, the number of unique track_ids matched the total number of entries in our dataset, indicating that each track possesses a distinct identifier.

We decided to re-run descriptive statistics on this new dataset to assess if there have been substantial changes in terms of distribution and relationships between variables. The only significant changes were observed in the top 5 artists and top 10 genres. Nevertheless, these modifications were expected since the rankings were influenced by counts within the dataset, and the removal of duplicates likely impacted their positions in the rankings.
![new_top_5_artists.png](Plots%2Fnew_top_5_artists.png)

![new_top_10_genres.png](Plots%2Fnew_top_10_genres.png)

 What is worth mentioning, however, is that the distribution of the new 'popularity' column remained the same as the original 'popularity' column. This suggests that our pre-processing efforts have maintained the overall distribution of popularity scores, meaning that we were successful in preserving the essential characteristics of the original data while addressing the issue of multiple values for the same track IDs. It is a positive indication that our pre-processing steps have been effective in preserving the integrity of the data and ensuring consistency in the popularity scores.
![new_distribution_popularity.png](Plots%2Fnew_distribution_popularity.png)

### 2.4 Splitting the Dataset, Encoding and Scaling

We started with a split of 80% in training set and 20% in test set.

After the splitting we proceeded with the encoding of the categorical variables.
We used two different types of encoding techniques: Multi Label Binarizer and Leave One Out encoding.

- MultiLabelBinarizer is an encoding technique similar to One Hot encoding but specifically designed for data with multiple labels. In our dataset, the "genre" variable consists of a list of genres associated with each song, making simple One Hot encoding inadequate. By using MultiLabelBinarizer, we were able to encode all the genres in the list, preserving all the relevant information.
- For the "artists" and "album_name" variables, we opted for Leave One Out encoding. Initially, we considered One Hot encoding, but due to the high cardinality of both variables, it would have resulted in a large number of encoded variables, leading to computational inefficiency. We also considered label encoding; however, since there was no apparent order in the data, we were hesitant to assign arbitrary numeric values. Therefore, Leave One Out encoding was chosen as it strikes a balance between information preservation and computational feasibility, allowing us to encode the categorical variables effectively without losing significant details. Leave One Out encoding operates similarly to target encoding, where each category level of a categorical variable is replaced with the mean value of the target variable (in this case, the mean popularity). However, Leave One Out encoding excludes the value in the current row when calculating the mean. This technique effectively reduces the risk of overfitting compared to standard target encoding, as it introduces a regularization effect. By utilizing Leave One Out encoding, we were able to encode the "artists" and "album_name" variables effectively, leveraging the regularization effect to reduce overfitting while still capturing the important information within the categorical variables.

### 2.5 Clustering

We performed K-Means clustering on both the entire dataset and the training set. In both cases, we focused solely on song features, excluding the "popularity" variable, as we considered it not to be a defining characteristic of a song.

For the clustering analysis, we utilized 114 clusters, which corresponded to the unique values of the "track_genre" variable. Our objective was to investigate whether genre could be objectively measurable based on the clustering results.

Following the clustering process, we added a new variable called "cluster" to both datasets. This variable indicated the cluster to which each data point belonged. Our intention was to examine the distribution of songs across the clusters and assess any potential imbalances.

To evaluate the quality of the clustering, we employed the Silhouette score. This metric allowed us to assess the coherence and distinctiveness of the clusters generated by the algorithm

### 2.6 Regression

In our regression analysis, we explored multiple algorithms and assessed their performance using key metrics such as R-Squared and Root Mean Square Error (RMSE).

We began with a linear regression baseline model and then progressed to a Decision Tree Regressor. From there, tried ensemble algorithms. Our initial ensemble models included a Random Forest Regressor and a Gradient Boosting Regressor. Additionally, we experimented with a Voting Regressor and a Stacking algorithm.

To optimize the Decision Tree, Random Forest, and Gradient Boosting models, we utilized grid or random search to identify the best combination of hyperparameters.

For the models we considered most significant, we visualized plots representing residuals and predicted values compared to the actual values, so that we could have a better understanding of the models' predictive capabilities.

In the subsequent section, we will delve deeper into the rationale behind our algorithm and metric choices, as well as provide insights into the implementation details.

## 3. Experimental Design

### 3.1 Clustering

The Silhouette Score was chosen as a metric for evaluating the quality of the clustering because it provides insights into the cohesion and separation of data points within each cluster. It allows for easy interpretation and comparison of different clustering solutions. The Silhouette Score ranges from -1 to 1, where -1 indicates poor clustering with data points assigned to incorrect clusters, 0 suggests overlapping or poorly separated clusters, and 1 signifies well-clustered data points with distinct and cohesive clusters.

Initially, we applied clustering to the full dataset by scaling the variables and fitting the clustering algorithm with the chosen number of clusters (K). Upon analyzing the resulting "cluster" variable, we observed that while some clusters had a high number of data points, many clusters contained fewer than 10 data points. Despite this imbalance, the Silhouette Score indicated moderate quality clustering, although the clusters were not entirely distinct.

To explore the possibility of improving clustering quality, we wondered if using a smaller sample might yield better results. We repeated the same steps using the X_train dataframe, excluding unnecessary variables and fitting the clustering algorithm. However, the clusters exhibited the same pattern as before since undersampling exacerbated the issue of low data points in certain clusters.In retrospect, it was unrealistic to expect this issue to be resolved by reducing the sample size. As anticipated, undersampling further decreased the clustering quality, as evidenced by a slightly lower Silhouette Score.

### 3.2 Regression

To evaluate our regression models, we selected two key metrics: R-Squared and Root Mean Square Error (RMSE).

- R-Squared was chosen for its straightforward interpretation. It ranges from 0 to 1, where 0 indicates that the model explains none of the variability in the data, and 1 indicates that the model explains all the variability. In this way, this metric provides an intuitive measure of the regression model's explanatory power. Additionally, R-Squared facilitates the comparison of different models, as a higher R-Squared value suggests a better fit to the data. However, R-Squared does have its limitations. It can be influenced by the number of independent variables and may give misleading interpretations in the presence of multicollinearity or violations of other model assumptions. Therefore, to gain a more comprehensive understanding of model performance, we complemented R-Squared with the Root Mean Square Error.
- The RMSE metric was considered suitable for our analysis because it shares the same unit of measurement as the dependent variable, in this case, "popularity." This allows for easy interpretation of the error metric within the context of the problem. RMSE provides a direct understanding of the average prediction error in the same units as the target variable.

By utilizing both R-Squared and RMSE, we aimed to gain insights into the explanatory power of the models (R-Squared) as well as the average prediction error (RMSE) in relation to the "popularity" variable. This combination of metrics provides a more comprehensive evaluation of the regression models' performance.

#### 3.2.1 Linear Regression

We started by implementing a Linear Regression model as a baseline.

The Linear Regression model is a simple and computationally efficient algorithm with easy interpretability. By comparing the results of more advanced models to the Linear Regression baseline, we can assess whether the additional complexity of the advanced models leads to significant improvements.

The Linear Regression model produced moderately good results, avoiding overfitting and providing a solid baseline. However, the model's assumptions of linearity and its simplicity prompted us to explore other, more complex algorithms.

#### 3.2.2 Decision Tree Regressor

After Linear Regression, we experimented with a Decision Tree Regressor model.

Decision trees are capable of capturing non-linear relationships and interactions between features without imposing specific functional forms. Compared to more complex approaches, decision trees offer greater interpretability and computational efficiency. However, decision trees may struggle to capture certain types of relationships that require more sophisticated modeling techniques, which could explain our relatively low results despite using grid search for hyperparameter tuning.

#### 3.2.3 Random Forest Regressor

Considering the limitations of the decision tree, we opted for a more complex algorithm, the Random Forest Regressor.

Random Forests combine multiple decision trees to make predictions and, for this reason, they often exhibit superior predictive performance compared to individual decision trees. Random Forests are generally less prone to overfitting and handle large datasets effectively. Moreover, they excel at capturing complex relationships that decision trees might fail to capture. Given the poor performance of the decision tree, Random Forest was a suitable choice.

Through a computationally expensive grid search, we identified the best hyperparameters and used them to fit the Random Forest model. Although the model's performance improved, we observed signs of overfitting.

To gain insights about the importance of different features for predictions, we plotted the feature importance derived from the Random Forest model. The analysis revealed that the "album_name" variable held the highest importance, closely followed by the "artists" variable. The feature importance analysis regarding the "album_name" variable confirmed our belief that it was essential to consider not only songs from the original albums but also those present in playlists. This finding indicated that the inclusion of songs in a specific album played a significant role in determining the popularity of a song. Therefore, taking into account both the original albums and the playlists proved to be crucial in accurately analyzing popularity scores of songs.

#### 3.2.4 Gradient Boosting Regressor

Due to the overfitting observed in the Random Forest model, we decided to explore a different type of ensemble learning algorithm, namely Gradient Boosting.

Gradient Boosting has generally high predictive performance and handles non-linear relationships effectively. Additionally, it is known to be more robust against overfitting. However, it is important to note that Gradient Boosting can be computationally expensive, especially with large datasets and complex models. As a result, both the random search (replaced the grid search due to its long execution time) and the model fitting process took a significant amount of time.

Despite the computational cost, the optimized Gradient Boosting model yielded good results, although some overfitting was still observed.

#### 3.2.5 Voting Regressor

To address the overfitting issue encountered in the Random Forest model while maintaining good performance, we decided to combine the Random Forest and Linear Regression models using a Voting Regressor.

By leveraging the different strengths of the two models, the Voting Regressor allowed us to take advantage of Linear Regression's effectiveness in capturing approximately linear relationships and Random Forest's ability to handle non-linear relationships and interactions.

The Voting Regressor significantly improved predictive performance, and importantly, mitigated the overfitting issue observed in the Random Forest model.

#### 3.2.6 Stacking Regressor

Considering the success of the Voting Regressor, we further explored the potential improvement of performance using the Stacking Regressor.

Similar to the Voting Regressor, the Stacking Regressor combines predictions from multiple models, including Random Forest and Linear Regression, to capture different aspects of the underlying relationships in the data. This approach offers the potential for enhanced predictive performance compared to the Voting Regressor, as the combination of models is done in a more sophisticated manner.

Although the Stacking Regressor exhibited good performance with a low RMSE, there was still a slight amount of overfitting observed. However, the model fitting process was relatively slow, making it less preferable in terms of efficiency.

## 4. Results
### 4.1 Clustering

After performing the clustering analysis, we obtained a silhouette score of 0.520 with the full dataset.
It looks like the clustering is of a moderate quality, and there might still be some overlapping clusters, but overall the clusters seem to be delineated.

However, when we tried the clustering on the X_train dataframe, the silhouette score was of 0.517, which still suggests a moderate level of clustering quality, albeit lower. 

Obtaining a non-optimal result with clustering for inferring the genre of a track is not surprising because clustering may not be the best choice for this specific task. There are several reasons why clustering may not be optimal for inferring the genre of a track:
- Ambiguity of musical genres: The concept of musical genre is often subjective and multidimensional. Tracks can incorporate elements from multiple genres or may be difficult to categorize into a single genre. This can lead to overlaps or confusion in the clusters generated by clustering.
- Complex musical characteristics: The musical features that influence the genre can be intricate and nuanced. Clustering algorithms like K-means may struggle to capture the subtle variations and complexities present in music that determine genre categorization.
- Contextual factors: Genre classification often requires contextual understanding and domain knowledge. Features like lyrics, artist information, cultural background, and historical context play significant roles in determining the genre of a track. Clustering algorithms may not take these contextual factors into account, leading to suboptimal results.

For this reason, we believe that alternative approaches could be more suitable. There are other machine learning techniques, such as supervised classification algorithms (e.g., decision trees, random forests, or support vector machines) that are commonly used for genre classification tasks. These methods leverage labeled training data and can offer better accuracy and interpretability compared to unsupervised clustering algorithms.

In summary, while clustering can provide insights into patterns and similarities in music, it may not be the most suitable approach for inferring the genre of a track due to the subjective nature of genres, complex musical characteristics, and the need for contextual understanding. Other approaches that consider labeled data and incorporate domain knowledge may yield more accurate results for genre classification tasks.

### 4.2 Regression

![results.png](Plots%2Fresults.png)

#### 4.2.1 Linear Regression
The linear regression model demonstrates moderate performance in predicting the target variable. The R2 value indicates the proportion of the variance in the dependent variable (target variable) that is explained by the linear regression model. In this case, the model explains approximately 68.9% of the variance in the training data and 65.1% of the variance in the test data. The R2 values are reasonably close, suggesting that the model generalizes well to unseen data.

The RMSE values of 11.47 for the train set and 12.14 for the test set indicate that, on average, the predicted values deviate from the actual values by approximately 11.47 and 12.14 units, respectively. The lower the rMSE, the better the model's predictive accuracy. While the rMSE values are not exceptionally low, they suggest a reasonable level of predictive performance. Despite the fact that the test set RMSE is slightly higher than the training set RMSE, the difference is minimal. Thus, it looks like the model is solid against overfitting.

Moreover, from the plot we see that the predicted values follow moderately well the distribution of the actual ones, without any particularly high peaks.

![distr-lr.png](Plots%2Fdistr-lr.png)

Overall, the linear regression model is providing a fair fit to the data, explaining a decent amount of the variance. However, there is still room for improvement, as indicated by the moderate R2 and RMSE values.

#### 4.2.2 Decision Tree Regressor
The decision tree regressor model shows moderate performance in predicting the target variable.

The R2 value of 0.58 for the train set indicates that approximately 58.9% of the variance in the target variable can be explained by the decision tree model's predictions. Similarly, the R2 value of 0.542 for the test set suggests that around 54.2% of the variance can be explained, indicating a moderate to low level of fit to the test data. The R2 values are reasonably close, suggesting that the model generalizes fairly well to unseen data.

The RMSE values of 13.22 for the train set and 13.91 for the test set indicate that, on average, the predicted values deviate from the actual values by approximately 13.22 and 13.91 units, respectively. These values provide an estimate of the average prediction error. Lower rMSE values indicate better predictive accuracy, so these moderate rMSE values suggest that the model does not predict a very accurate popularity. However, train and test RMSE are very similar, meaning that the model looks like it robust against overfitting.

Moreover, from the distribution plot, we see a particularly high peak around 40, which is probably the reason why the performance is so low. 
It seems like the model might have captured some bias in the data which led to these not very good results.
![dc-distr.png](Plots%2Fdc-distr.png)

Overall, the decision tree regressor model provides a reasonable fit to the data, but there may be room for improvement in terms of increasing the R2 and reducing the rMSE values.

#### 4.2.3 Random Forest Regressor
The random forest regressor model shows good performance in predicting the target variable.

In this case, the model explains approximately 81.1% of the variance in the training data and 73.3% of the variance in the test data. The R2 values are relatively close, although there is a difference which might indicate a potential overfitting in the model, which might not generalize particularly well to unseen data.

The RMSE on the training set is 8.96, while the RMSE on the test set is 10.63. The test set RMSE is slightly higher than the training set RMSE, which suggests that the model may be slightly overfitting the training data.

When we visualized the plot, we noticed that it still kept the trend of the decision tree for what regards the distribution of the predicted and actual values, albeit in a smaller way.
Indeed, we can see that there still is a peak in the area around 40, indicating that the model might have captured some of the noise that the decision tree also had.

![distr-rf.png](Plots%2Fdistr-rf.png)

However, overall, the random forest regressor model demonstrates strong performance, as indicated by the high R2 values and relatively low RMSE values. It seems to be capturing the underlying patterns in the data and generalizing well to the test set. It is a reliable model for predicting the target variable.

#### 4.2.4 Gradient Boosting Regressor
The gradient boosting regressor model demonstrates good performance in predicting the target variable.

The R-squared value of 0.848 for the training set indicates that approximately 84.8% of the variance in the target variable can be explained by the model's predictions. Similarly, the R-squared value of 0.798 for the test set suggests that around 79.8% of the variance can be explained. Since values are close to 1, this indicates a good fit of the model to the data.

The RMSE values of 8.012 for the training set and 9.247 for the test set indicate that, on average, the predicted values deviate from the actual values by approximately 8.012 and 9.247 units, respectively. These values indicate that the gradient boosting regressor model is capturing a significant portion of the data's variability and providing reasonably accurate predictions. However, the test set RMSE is slightly higher than the training set RMSE, which suggests that the model may be slightly overfitting the training data.

However, the plot of the distribution of the values looks much better than the one of the previous models. It follows the distribution of the actual values much more closely, which is another good indication that the model is performing well.

![gb-distr.png](Plots%2Fgb-distr.png)

Overall, the gradient boosting regressor model demonstrates strong performance, as indicated by the high R-squared values and relatively low RMSE values. It effectively captures the underlying patterns in the data and provides accurate predictions for the target variable. It could be a reliable model for the given task.

#### 4.2.5 Voting Regressor
The voting regressor model demostrates strong performance in predicting the target variable.

The voting regressor shows relatively good performance in terms of the R2 values on both the training and test sets. It explains approximately 78.1% of the variance in the training data and 72.2% of the variance in the test data. This suggests that the model captures a substantial portion of the target variable's variability.

There is a notable difference between the training and test set performance, particularly in terms of RMSE. The training set RMSE is 9.62, while the test set RMSE is significantly lower at 3.29. The lower test set RMSE compared to the training set RMSE suggests that the model generalizes well and is able to make accurate predictions on new, unseen data. This is a good indication of the model's performance and suggests that it is effective in capturing the underlying patterns and relationships in the data.

Furthermore, it is interesting to notice how the plot of the distribution of the predicted and actual values, follows more the one of the linear regression, and there are no peaks at around 40 like there were in the random forest one.

![vot-distr.png](Plots%2Fvot-distr.png)

Overall, based on the RMSE values, the voting regressor seems to be performing well. It demonstrates a lower prediction error on the test set, indicating its ability to generalize to new data points.

#### 4.2.6 Stacking Regressor
The stacking regressor model shows a good performance in predicting the target variable.

In this case, the model explains approximately 80.9% of the variance in the training data and 73.8% of the variance in the test data. The R2 values suggest that the model performs well in explaining the variance in both sets.

The RMSE on the training set is 8.98, while the RMSE on the test set is 3.24. The test set RMSE is significantly lower than the training set RMSE, suggesting that the model generalizes well and is able to make accurate predictions on unseen data and that it is effective in capturing the underlying patterns and relationships in the data.

Additionally, the plot of the distribution of the predicted and actual values looks much more like the one of random forest, contrary to the voting regressor one. Indeed, we see that there is again the peak around 40.

![stack-distr.png](Plots%2Fstack-distr.png)

Overall, the stacking regressor demonstrates a good performance. It exhibits good predictive ability and explains a significant portion of the variance in both the training and test datasets.
Despite this, the model takes a long time to run, which makes it not the best choice, especially for larger datasets.

## 5. Conclusions

Our analysis aimed to understand the factors behind song popularity and identify the types of songs that appeal to different users. This knowledge can be utilized to develop better recommendation models, enhancing user engagement and retention on the platform.

Regarding track genre clustering, it is important to note that while clustering can uncover patterns and similarities in music, it may not be the optimal approach for inferring track genres. Genres are subjective, complex, and require contextual understanding, making it challenging to accurately classify tracks through clustering alone. Moreover, it is not clear how Spotify assigns the genres to a song, although, judging by the multiple genres assigned to some of the songs, it looks like it might be due to factors such as the users preferences, or the placement of a song in different kinds of playlists. 
Alternative methods that might help with inferring the genre, however, incorporate labeled data and domain knowledge, which are likely to yield more precise results.

Regarding song popularity, the voting regressor and the gradient boosting models performed well. However, it is important to acknowledge that predicting the popularity of a song with algorithmic models is not always highly accurate, and will possibly never be. Indeed, song popularity is influenced by various factors that are not always measurable, such as cultural trends and individual preferences, but also advertising strategies of the individual artists and their managers, as well as the social and political landscape. 
Consequently, we recommend that music producers and artists focus on understanding their target audience and tailoring their music to meet their preferences and interests. By conducting market research, analyzing listener feedback, and engaging with their fan base, artists can gain valuable insights into the specific characteristics and elements that resonate with their audience. This could include factors such as exploring new genres or sub-genres that are gaining popularity, incorporating cultural or regional influences into their music, or experimenting with innovative production techniques. By staying connected to their fans and adapting their creative approach accordingly, artists can increase their chances of creating music that truly connects with and engages their target audience; moreover, analyzing user preferences and leveraging the insights gained from the analysis can guide their decision-making process and increase the likelihood of creating music that resonates with their target audience.

While our work has provided valuable insights into the factors influencing song popularity and patterns in user preferences, there are still some questions that may not be fully answered. For example, our analysis primarily focused on the audio features and general characteristics of songs, but there could be additional factors such as lyrics, artist popularity, or cultural context that play a significant role in determining song popularity. Exploring these factors in future research could provide a more comprehensive understanding of the dynamics behind song popularity. Additionally, incorporating user-specific data, such as user demographics or listening history, could further enhance the accuracy of recommendation models. Furthermore, conducting user surveys or feedback analysis could provide deeper insights into the subjective aspects of song preference. These directions of future work can contribute to refining recommendation algorithms and personalizing the music streaming experience for users.
