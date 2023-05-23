# POPULARITY SCORE OF MUSIC

## Costanza Placanica - Giulia Ghislanzoni



## 1. Introduction
The music streaming industry is highly competitive, with companies competing to attract and retain users by providing a wide range of songs and personalized recommendations. To gain a competitive edge, a large music streaming service has tasked us to conduct an exploratory data analysis on a comprehensive dataset of songs, including information on their audio features, popularity, and track, album and artist details. By analyzing this dataset, we aim to uncover insights into the characteristics of popular songs and identify patterns that can inform the development of more accurate recommendation models. Ultimately, our analysis can help the company enhance user engagement and retention by providing tailored music recommendations that align with users' preferences.

Moreover, using different regression techniques, we aim to provide the company with a reliable and efficient predictor for the popularity of the songs, in order to help the business on investing their resources strategically, such as signing artists, promoting certain songs, or securing licensing deals. This can lead to increased sales, streaming numbers, concert ticket sales, merchandise purchases, and other revenue-generating opportunities.

## 2. Methods
![Flowchart.png](Plots%2FFlowchart.png)

### 2.1 Environment
The deliverable is divided into a MAIN.ipynb file, which is a file Jupyter Notebook where we wrote all the code, and this file README.md, where we wrote the report.

In addition, we will provide all the plots used in this report in the folder, in order to be able to be displayed in this file accordingly.

For dealing with the data, we used the libraries Pandas and Numpy, while for the visualizations, we used both the library Seaborn and the library Matplotlib.
Finally, we mostly used the library Sklearn for all the clustering and regression models, as well as the metrics we used. 

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
- 
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

Moreover, we decided to explore which where the top 10 track genres in our dataset. Therefore, we first calculated the average popularity for each track_genre and get the first 10. Then, we plotted a bar plot showing the top 10 genres ranked by average popularity. Theis analysis reveals that the audience has diverse and eclectic music tastes. Pop-film is the most popular genre, followed by K-Pop, chill music, and sad music.  Indian music and anime music also rank high, showing a preference for traditional and culturally significant music and music associated with Japanese media. What is surprising is the presence of Sertanejo music, a popular genre in Brazil, that also seems to have a significant following among the  audience.
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
We used two different types of encoding techniques: multi label Binarizer and leave on out encoding.

Multi Label Binarizer is a type of encoding similar to one hot encoding, but it is used in data with multiple labels as it enables to encode multiple labels per instance. Indeed, the variable genre in our dataset is comprised of a list of genres associated to every song, which meant that a simple one hot encoding was not going to be enough. However, with Multi Label Binarizer, we were able to encode all the genres in the list, so that we could keep all the information.

For the other two categorical variables, artists and album_name, we decided to use leave one out encoding. 
At first we thought about one hot encoding, but it would have been too computationally expensive to have so many variables, as both artists and album_name had a high cardinality; we also thought about label encoding, we were not comfortable with assigning an order to our data that did not apparently have one.
Therefore, we settled for a leave one out encoding. Leave one out encoding works like a target encoding, meaning that replaces the value of each category level of a categorical variable with their respective mean of the target variable (thus we used the mean of the popularity), although is does not include the value in the current row. In this way, while target encoding has limitations because it might be prone to data leakage and overfitting, leave one out has a regularisation effect, which reduces the risk of overfitting as it makes the encoding less dependent on the target variable.  

### 2.5 Clustering

Regarding K-Means clustering, we did it on both the entire dataset and then on only the training set.
In both cases, we only considered the song features, also excluding the popularity, as in our opinion it does not really constitute a feature of a song. 

Moreover, we used 114 clusters, which are the unique values of the variable track_genre, in order to understand whether the genre could have been an objectively measurable variable.

After the clustering, we added another variable in both datasets, called "cluster", where we reported for each data point the cluster that point belonged to. Our aim was understanding the distribution of songs in the clusters, and whether they might be unbalanced.

Finally, we used the Silhouette score to assess the performance of the clustering.

### 2.6 Regression

For what regards the regression models, we proceeded with trying many algorithms and evaluating them based on metrics such as R-Squared and Root Mean Square Error.

Thus, we first started with a linear regression baseline model. We moved on to a Decision Tree Regressor and then to ensemble algorithms. 
In particular we first tried a Random Forest Regressor and a Gradient Boosting Regressor. Then, we experimented with a Voting Regressor as well as a Stacking algorithm.
Decision Tree, Random Forest and Gradient Boosting were optimized through a grid search, in order to find the best parameters.
 
For the models which we considered to be more important, we visualized plots regarding the residuals and the predicted values compared to the real values, so that we could understand better what the models were predicting.

In the next section, we will expand more on why we chose those algorithms and metrics in particular, as well as how we implemented them. 

## 3. Experimental Design

### 3.1 Clustering

First of all, we decided on the Silhouette Score as a metric because it provides a measure of how well the data points within each cluster are separated from other clusters, indicating the overall cohesion and separation of the clusters.
Furthermore, the Silhouette Score is easily interpretable, which makes it very good for comparing different clustering solutions as in our case. 

Its range of values goes from -1 to 1, with -1 implying that the data points are incorrectly clustered or have been assigned to the wrong clusters, suggesting that the clustering solution is poor and the data points would be better assigned to a different cluster or treated as outliers.
A silhouette score of 0 indicates that there are overlapping or poorly separated clusters, and that the data points may not be clearly assigned to their respective clusters, leading to some degree of ambiguity or overlap between clusters.
Finally, a silhouette score of 1 indicates that the data points are well-clustered, with clear separation between clusters and high cohesion within clusters, meaning that the clustering solution is appropriate and the clusters are distinct.

Therefore, we first tried the clustering on the full dataset: we first scaled the variables, then we fitted the clustering with the chosen K.

We then looked at the new variable "cluster", and noticed that, while there were many clusters with a very high number of values, there werwe also many that had less than 10 data points.
However, when we calculated the Silhouette score, it looked like the clusters were of a moderate quality, although not entirely distinct.

For these reasons, we were curious as to whether using a smaller sample might help in getting a higher clustering quality. 
We repeated the exact same steps as before but using the X_train dataframe, removing the variables we did not need and then fitting the clustering. 

Of course, the clusters exhibited the same pattern as before, because there were still many with a low number of data points. In retrospect, we should not have expected this to change, as reducing the sample was only going to exacerbate this problem.
Indeed, when we computed the Silhouette score, it was clear that not only had undersampling not improved the clustering, it had even slightly worsened it.


### 3.2 Regression

As mentioned above, in order to evaluate out models, we chose the R-Squared and the Root Mean Square Error.

The reasons why we have decided to use the R-Squared are two-fold: first, for its straightforward interpretation, as it ranges from 0 to 1, where 0 indicates that the model explains none of the variability in the data, and 1 indicates that the model explains all the variability. In this way, it provides a simple and intuitive way to communicate the explanatory power of the regression model.
Second, because R-squared allows for easy comparison of different models, since when comparing multiple regression models, the model with a higher R-squared value is generally considered to have a better fit to the data.

However, the R-Squared still does have its limitations, as it can be influenced by the number of independent variables, and its interpretation may be misleading in the presence of multicollinearity or other model assumptions violations. Therefore, we thought it best to use it in conjunction with another metric: the Root Mean Square Error. 

This metric in our case can be a very suitable choice, because the RMSE has the same unit of measurement as the dependent variable, popularity, which makes it easier to interpret the error metric in the context of the problem and provides a direct understanding of the average prediction error in the same units as the target variable.

#### 3.2.1 Linear Regression
We first decided to try with a linear regression as baseline, since it is a simple and computationally efficient algorithm which is easily interpretable. Moreover, by comparing the results of more advanced models to those of linear regression, we can assess whether the additional complexity of the advanced models leads to significant improvements.
Linear regression gave moderately good results, resisting overfitting and providing a good baseline. However, its assumptions of linearity, and the perhaps too simple model led us to try with other, more complex algorithms. 

#### 3.2.2 Decision Tree Regressor
Therefore, after Linear Regression, we tried with a Decision Tree Regressor model.
We decided first on a Decision Tree because they can capture non-linear relationships and interactions between features without explicitly assuming a specific functional form. Moreover, compared to other more complex approaches they are more interpretable and more computationally efficient. 
However, decision trees may struggle with capturing certain types of relationships that require more complex modeling approaches, which is probably why our results, despite the grid search, were so low. 

#### 3.2.3 Random Forest Regressor
For this reason, we thought about a more complex algorithm, Random Forest, which combines multiple decision trees to make predictions. 
Indeed, Random Forests often provide better predictive performance compared to individual decision trees, by aggregating the predictions from multiple trees. In general, they are less prone to overfitting compared to decision trees and they can handle large datasets very well. Finally, they are able to capture more complex relationships which decision trees might not be able to grasp, and thus we decided to use it, especially after the poor performance of the decision tree. 

After running a grid search, which was very computationally expensive, we found the best hyperparameters and used them to fit the model. 
However, despite the improvement in the performance, it looked like there was some overfitting in the data.

From random forest, we also plotted the feature importance, which we used to better understand the variables which were more relevant for the predictions.
It looked like the variable album_name was the most important one, directly followed by artists.

#### 3.2.4 Gradient Boosting Regressor
Because of the overfitting of random forest, we wanted to try a different kind of ensemble learning, and we settled on Gradient Boosting. 
In general, gradient boosting has a high predictive performance and handles very well non-linear relationships. Moreover, it should be more robust against overfitting.

Despite this, gradient boosting is very computationally expensive, especially with large datasets and complex models, which made both the random search (we had to change to random search as the grid search was taking too long), and the model fitting take an important amount of time. 

Nevertheless, after the optimization, the results were pretty good, although it still looked a bit like it was overfitting.

#### 3.2.5 Voting Regressor
We noticed that although random forest was overfitting, the results were good. We needed a way to reduce the overfitting (and to find a model that was less computationally expensive than Gradient Boosting), and we realized that we could have combined the Random Forest model with the Linear Regression one, in a Voting Regressor.

Indeed, because both models are very different from one another, by combining them in a Voting Regressor, we could leverage the strengths of each model and potentially improve overall prediction performance. Linear regression might be effective when the relationship between the features and the target variable is approximately linear, while random forest can capture non-linear relationships and interactions.

Indeed, the predictive performances were much better, and, most importantly, the model was not overfitting anymore.

#### 3.2.6 Stacking Regressor
However, because the voting regressor had produced good results, we were interested to find out whether another method, Stacking, would also improve the performances.

Again, we used random forest and linear regression, and we tried to improve the performance without sacrificing too much time. 

Stacking allows to combine the strengths of different models. As in voting regressor, by combining the predictions of Linear Regression and Random Forest, we can potentially capture different aspects of the underlying relationships in the data, leading to improved overall performance.

Moreover, Stacking has the potential to provide better predictive performance compared to Voting, by combining the predictions of multiple models in a more sophisticated way.

In the end, the performance of the stacking regressor was good, although it looked like there was still a small amount of overfitting, despite the RMSE giving very good results. However, the model fitting was very slow, making it not the best choice. 

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
