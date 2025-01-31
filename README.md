  # **Marketing Campaign Analysis**

  ## Problem Definition

### The Context
Why is this problem important to solve?

Customer segmentation is crucial because it allows businesses to better understand their customers, tailoring marketing strategies to different customer groups based on unique characteristics. This leads to improved customer engagement, increased sales, and optimized use of resources. Segmented campaigns can significantly boost click rates and overall revenue by targeting personalized communications and offerings.

### The Objective
What is the intended goal?

The objective is to utilize unsupervised learning techniques such as dimensionality reduction and clustering to identify the best possible customer segments from the given dataset. This will enable more effective and efficient marketing strategies, ultimately enhancing return on investment (ROI).

### The Key Questions
What are the key questions that need to be answered?

1. What are the distinct customer segments within the dataset?
2. What characteristics define each customer segment?
3. How can these segments be leveraged to optimize marketing strategies?
4. Which clustering method provides the most meaningful and actionable segments?

### The Problem Formulation
What is it that we are trying to solve using data science?

We aim to solve the problem of understanding customer behavior and characteristics by dividing the customer dataset into meaningful segments. This involves analyzing customer profiles, campaign conversion rates, and engagement with marketing channels to create segments that can be targeted with tailored marketing strategies. The goal is to enhance marketing efficiency and improve ROI by effectively utilizing customer data.


------------------------------
## **Data Dictionary**
------------------------------

The dataset contains the following features:

1. ID: Unique ID of each customer
2. Year_Birth: Customer’s year of birth
3. Education: Customer's level of education
4. Marital_Status: Customer's marital status
5. Kidhome: Number of small children in customer's household
6. Teenhome: Number of teenagers in customer's household
7. Income: Customer's yearly household income in USD
8. Recency: Number of days since the last purchase
9. Dt_Customer: Date of customer's enrollment with the company
10. MntFishProducts: The amount spent on fish products in the last 2 years
11. MntMeatProducts: The amount spent on meat products in the last 2 years
12. MntFruits: The amount spent on fruits products in the last 2 years
13. MntSweetProducts: Amount spent on sweet products in the last 2 years
14. MntWines: The amount spent on wine products in the last 2 years
15. MntGoldProds: The amount spent on gold products in the last 2 years
16. NumDealsPurchases: Number of purchases made with discount
17. NumCatalogPurchases: Number of purchases made using a catalog (buying goods to be shipped through the mail)
18. NumStorePurchases: Number of purchases made directly in stores
19. NumWebPurchases: Number of purchases made through the company's website
20. NumWebVisitsMonth: Number of visits to the company's website in the last month
21. AcceptedCmp1: 1 if customer accepted the offer in the first campaign, 0 otherwise
22. AcceptedCmp2: 1 if customer accepted the offer in the second campaign, 0 otherwise
23. AcceptedCmp3: 1 if customer accepted the offer in the third campaign, 0 otherwise
24. AcceptedCmp4: 1 if customer accepted the offer in the fourth campaign, 0 otherwise
25. AcceptedCmp5: 1 if customer accepted the offer in the fifth campaign, 0 otherwise
26. Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
27. Complain: 1 If the customer complained in the last 2 years, 0 otherwise

**Assumption:** The data is collected in the year 2016.

- Histogram of Income (initial)

![income_histogram_before](https://github.com/user-attachments/assets/4c32d595-1e60-4aec-8af7-593b512bac4c)

- Histogram of Income (after outlier was dropped)

![income_histogram_after](https://github.com/user-attachments/assets/b1dcc7fd-3239-4468-896c-f4ba5020c760)

- Histograms for other variables

![wine_histogram](https://github.com/user-attachments/assets/81974139-0d39-49a7-b8d1-04f51be7f211)

![fruits_histogram](https://github.com/user-attachments/assets/20626828-e56f-4b0b-8116-9497556277d3)

![meat_histogram](https://github.com/user-attachments/assets/82b500e0-c050-4f21-82b3-b08bf5fff110)

![products_histogram](https://github.com/user-attachments/assets/f72d2637-caae-4e23-9737-8fd26804342d)

![sweet_histogram](https://github.com/user-attachments/assets/c7962812-6e20-490d-b9bf-9fa0a073fb16)

![gold_histogram](https://github.com/user-attachments/assets/ca99abbb-3bff-4cb0-a8a7-eecd683d7690)

- Bar Plot of Marital Status

![marital_status_barplot](https://github.com/user-attachments/assets/ca472720-c89c-46f5-be82-16ddbd46a433)

- Heatmap (initial)

![heatmap](https://github.com/user-attachments/assets/ba86a593-59fa-4469-ad41-62fb8d3b6dcc)

- Education vs Income Barplot

![education_income_barplot](https://github.com/user-attachments/assets/fe5b1f59-744c-4172-9cd1-5692e07c76f6)

- Marital Status vs Income Barplot

![marital_status_income_barplot](https://github.com/user-attachments/assets/eb048a18-c03e-4f94-bcc5-1eaeae00bdf7)

- Number of Kids at Home vs Income

![numberofkids_income_barplot](https://github.com/user-attachments/assets/a34e60b7-16e6-4c95-a9a3-90d19cb3b660)

- Crosstab of Marital Status vs Number of Kids at Home

![crosstab_maritalstatus_numofkids](https://github.com/user-attachments/assets/9ec454e8-31e5-4484-8253-3bf87b262c00)

- Histogram of Age

![age_histogram](https://github.com/user-attachments/assets/48eb41d0-70f0-4b4a-af05-8d8bb2d3a27b)

- Histogram of Amount per Purchase

![amnoutperpurchase_distribution](https://github.com/user-attachments/assets/5ee3b712-ed04-473a-a0d8-d6a77e6d4ad4)

- Scatterplot of Income vs Expenses

- ![expenses_income_scatterplot](https://github.com/user-attachments/assets/045a9c7e-78ba-4fc0-9c94-356070737bec)

- Family Size vs Income

![familysize_income_barplot](https://github.com/user-attachments/assets/e64ddb2b-7567-4353-9443-65be2ea3423d)

- Heatmap (New)

![heatmap_after](https://github.com/user-attachments/assets/118d1e0d-8ba8-458b-a7be-32cb691cb065)

- Scatterplot of t-SNE Clusters

![t-SNE](https://github.com/user-attachments/assets/798d8a29-9661-49fa-9611-0cf181923e3a)

- Scatterplot of PCA Application

![pca_application](https://github.com/user-attachments/assets/6e29d07b-6abc-4170-8a73-ecd0229a65a8)

- KMeans Elbow Plot

![K-Means_Elbow_Curve](https://github.com/user-attachments/assets/3cf93772-7bb1-41af-96eb-2051d7e6d9e8)

- Scatterplot of KMeans (3 Clusters) 

![kmeans_pca_clustering_3clusters](https://github.com/user-attachments/assets/c2d89476-05e8-48ce-b173-768a6bf5afde)

- Boxplot of KMeans (3 Clusters)

![boxplot_kmeans_3clusters](https://github.com/user-attachments/assets/78221519-0fc8-49d9-b671-0e8bc37173fb)

- Scatterplot of KMeans (5 Clusters)

![kmeans_pca_clustering_5clusters](https://github.com/user-attachments/assets/206f05fa-fbca-4511-a1e5-f9dc7419e3d7)

- Boxplot of KMeans (5 Clusters)

![boxplot_kmeans_5clusters](https://github.com/user-attachments/assets/70a2422d-9887-4db0-bbaf-4ddb301e30fe)

- Scatterplot of KMedoids (5 Clusters)

![kmedoids_clustering_5clusters](https://github.com/user-attachments/assets/5c52784c-5332-4cb7-8791-e5bb7f58b48d)

- Boxplot of KMedoids (5 Clusters)

![boxplot_kmedoids_5clusters](https://github.com/user-attachments/assets/2beb9828-03c0-48b6-9a12-8dd6430fc440)

- Scatterplot of Hierarchical Clustering (3 segments)

![hierarchical_clustering_3clusters](https://github.com/user-attachments/assets/e41b09af-7e53-46c2-b3a8-df8ccbf23608)

- Boxplot of Hierarchical Clustering (3 segments)

![boxplot_hierarchical_clustering_3clusters](https://github.com/user-attachments/assets/dc516143-7d2d-424a-ac86-32a44c9bb6e2)


- Boxplot of DBSCAN

![dbscan_pca](https://github.com/user-attachments/assets/11f95285-9adc-4773-9227-f3bb4482e557)

- Scatterplot of Gaussian Mixture Model (5 Clusters)

![gmmcluster](https://github.com/user-attachments/assets/5d7ebd7d-6f71-416b-abb9-5949d04919ab)

- Boxplot of Gaussian Mixture Model (5 Clusters)

![boxplot_gmm](https://github.com/user-attachments/assets/add26556-bd4e-44e0-828a-b4946e35a6f3)


## **Conclusion and Recommendations**

#### 1. Comparison of Various Techniques and Their Relative Performance

We evaluated multiple clustering techniques based on the silhouette score, which measures how similar an object is to its own cluster compared to other clusters. Here’s a comparison:

1. **K-Means**:
   - For (n_clusters = 3): Silhouette score = 0.2694
   - For (n_clusters = 4): Silhouette score = 0.2547
   - For (n_clusters = 5): Silhouette score = 0.2337
   - For (n_clusters = 6): Silhouette score = 0.2186
   - **Best score**: K-Means with (n_clusters = 3), silhouette score = 0.2694

2. **K-Medoids**:
   - Silhouette score for (k = 5): 0.1192

3. **DBSCAN**:
   - (eps = 2), (min_samples = 6): Silhouette score = 0.1112
   - (eps = 2), (min_samples = 20): Silhouette score = 0.3385
   - (eps = 3), (min_samples = 6): Silhouette score = 0.2237
   - (eps = 3), (min_samples = 20): Silhouette score = 0.3382
   - **Best score**: DBSCAN with (eps = 2) and (min_samples = 20), silhouette score = 0.3385

4. **Gaussian Mixture Model (GMM)**:
   - (n_components = 5): Silhouette score = 0.1341

**Performance Summary**:
- **DBSCAN** with (eps} = 2) and (text{min_samples} = 20 ) had the highest silhouette score (0.3385).
- **K-Means** with (n_clusters = 3) provided a strong performance with a silhouette score of 0.2694.

**Scope for Improvement**:
- Further tuning of DBSCAN parameters (eps and min_samples) could potentially improve performance.
- Combining clustering methods or integrating domain-specific knowledge might yield better results.

#### 2. Refined Insights

- **DBSCAN** was highly effective in identifying well-separated clusters and managing noise points. This is beneficial for data sets with outliers or irregular cluster shapes.
- **K-Means** showed solid performance, especially with 3 clusters, indicating that the data might naturally group into three distinct segments.
- **GMM** provided a moderate performance, but offers the flexibility of soft clustering, where data points can belong to multiple clusters with different probabilities.

**Key Insights**:
- The data contains clear subgroups that can be identified with proper parameter tuning.
- Outlier detection is crucial for understanding the data's structure and should be integrated into the final solution.

#### 3. Proposal for the Final Solution Design

Based on the analysis, I propose adopting **DBSCAN** with (text{eps} = 2) and (text{min_samples} = 20) for the following reasons:

- **Highest Silhouette Score**: Achieved the best-defined clusters among the methods tested.
- **Flexibility**: Effectively handles outliers and finds arbitrarily shaped clusters, making it versatile for various data structures.
- **No Need for K**: Does not require specifying the number of clusters in advance, simplifying the clustering process.

**Implementation Steps**:
1. **Cluster the Data**: Use DBSCAN with the optimal parameters to cluster the data.
2. **Visualize and Interpret**: Create visualizations to understand the clusters and their characteristics.
3. **Validate**: Ensure the clusters make sense in the context of the problem using domain-specific knowledge.

This combination of flexibility, performance, and ease of use makes DBSCAN the best solution for your clustering needs.









