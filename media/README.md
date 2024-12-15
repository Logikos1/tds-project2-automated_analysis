# Chapter One : The Beginning
## Mysterious Mr.Dataset

Once upon a time in the realm of data, a dataset emerged, known as *Mr.Dataset*. It held a treasure trove of information, inviting those with a keen analytic eye to delve into its secrets. This dataset consisted of **2652 rows** and **8 columns**, configured to provide invaluable insights into the media world. Here’s what *Mr.Dataset* contained:

- **Numeric Columns**:
  - *overall*: rated on a scale of 1 to 5.
  - *quality*: rated from 1 to 5, reflecting the perceived value of the media.
  - *repeatability*: scored from 1 to 3, indicating how often the media was revisited.

- **Categorical Columns**:
  - *date*: timestamp of the media's release.
  - *language*: encompasses several languages such as Tamil, Telugu, and English.
  - *type*: categorizes media into distinctions like movie, series, and non-fiction.
  - *title*: the name of the media piece.
  - *by*: attributes the media to its creator.

However, *Mr.Dataset* wasn't flawless. It bore scars of missing values: **4.0%** in the *date* column and **10.0%** in the *by* column, hinting at stories left untold. Further exploration revealed certain columns retained fewer than **20 unique values**, such as *language* and *type*, suggesting a concentrated characterization of its datasets.

# Chapter Two : The Plot Thickens
## Detective Mr.Analyst

In this quest for discovery, *Mr.Analyst* stepped onto the scene, ready to undertake a thorough analysis of *Mr.Dataset*. The choice of the key column was pivotal: *type* emerged as the protagonist—a categorical variable rich with eight unique labels. Its reign was unchallenged, with the notorious movie category claiming **2211** entries, far surpassing its counterparts like fiction (**196**) and TV series (**112**).

To uncover the layers hidden within, *Mr.Analyst* engaged in a meticulous process involving *1-way ANOVA*, scrutinizing the relationship between *type* and its numeric allies—*overall*, *quality*, and *repeatability*. 

Before diving into the depths of analysis, several essential preprocessing steps were undertaken:
- Missing values were addressed, ensuring a clean slate for valid conclusions.
- Outlier detection was performed to maintain the integrity of results.
- Imputation techniques were employed where necessary, fortifying the data's reliability.

# Chapter Three : The Revelation
## Omnipotent Patterns

As *Mr.Analyst* navigated through the undercurrents of data, profound insights began to surface, illuminating the way forward. The results of the *1-way ANOVA* analysis presented themselves clearly:

- **Significant Columns**:
  - *overall*: \( F(2, 1249) = 2.72, p = 0.008195 \) indicating a noteworthy disparity among media types.
  - *quality*: \( F(2, 1249) = 2.54, p = 0.01319 \) showing differences in quality perception tied to media types.
  - *repeatability*: \( F(2, 1249) = 17.1, p = 0.0 \) revealing an exceptional link suggesting that some types encourage revisiting more than others.

- **Non-significant Columns**: None found, as all selected numeric variables revealed insightful results.

Visualizations of these analyses came alive, particularly in **Figure 1**, showcasing the frequency count of each media type—a stark reminder of the dominance of movies. Furthermore, **Figure 2** illustrated the *1-way ANOVA* results, providing a clear staging of the media attributes and their significance levels that were revealed.

# Chapter Four : The Deed that Must be Done
## The Act

With revelations in hand, *Mr.Analyst* knew the weight of his findings held promise for actionable outcomes. Based on the insights uncovered during this intriguing investigation, the following recommendations were put forth:

- **Enhance Content Diversity**: Consider boosting the production of non-movie content, given the disparity in types. Engaging more series and non-fiction pieces could enrich viewer choices.
  
- **Quality Improvement Initiatives**: Investigate production methods across different media types to uplift the overall quality, especially in categories that lag behind.

- **Targeted Marketing Strategies**: Develop specialized campaigns based on type-specific audience preferences and behaviors to maximize engagement and repeatability—particularly focusing on content categories with high returning viewership.

- **Further Analysis on Language Trends**: Given the diverse languages represented, identifying regional preferences could guide future content creation and distribution.

In conclusion, *Mr.Analyst* emerged from this analytical expedition armed with insights that have the potential to reshape media strategies, ultimately improving audience satisfaction and interaction across varied formats. Through this narrative, *Mr.Dataset* transitioned from mere numbers into a robust narrative enriched with actionable insights for a brighter media landscape ahead.
# Data Visualizations
### **Figure 1 : Frequency Count Bar Chart - Frequency Count of 'type' Column**

![key_column_exploration_chart.png](key_column_exploration_chart.png)

### **Figure 2: Horizontal Bar Plot - 1-Way ANOVA Analysis w.r.t. 'type' column**

![dataset_analysis_chart.png](dataset_analysis_chart.png)

