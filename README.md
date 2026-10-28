# Amazon Customer Sentiment Analysis

**Syntecxhub Virtual Internship Program — Data Analysis Track, Task 4**

> Intern: Maheen Irfan
> Domain: Data Analysis
> Program: Syntecxhub Virtual Internship

---

## Overview

This project applies Natural Language Processing (NLP) to classify the sentiment of 10,000 real Amazon product reviews as Positive, Negative, or Neutral, then validates the results against actual star ratings and extracts the key themes driving customer satisfaction and dissatisfaction.

## Dataset

**Source:** Amazon Fine Food Reviews (a well-known public benchmark dataset of real Amazon customer reviews).
- Total reviews analyzed: 10,000
- Fields used: review text, star rating (1–5), review date
- No missing values or duplicates in this sample

## Objective

As per the task brief, this project covers:
- Analyzing an Amazon product reviews dataset
- Cleaning and preprocessing review text data
- Performing sentiment classification (positive, negative, neutral)
- Identifying patterns in customer feedback and ratings
- Visualizing sentiment distribution and trends
- Generating insights for product improvement and customer satisfaction

## Key Results

| Metric | Value |
|---|---|
| Total Reviews | 10,000 |
| Positive | 88.92% |
| Negative | 9.05% |
| Neutral | 2.03% |
| Average Star Rating | 4.13 / 5 |
| Model Agreement with Star Ratings | 79.16% |

The **model agreement score** compares the NLP-predicted sentiment against the actual star rating (4–5★ = Positive, 1–2★ = Negative, 3★ = Neutral) to validate that the automated sentiment classification is a reliable proxy for real customer opinion.

### Most Common Words
- **Positive reviews:** taste, flavor, great, good, love, coffee
- **Negative reviews:** taste, coffee, disappointed, would (often "wouldn't buy again"), stale

## Project Structure

```
Syntecxhub_AmazonSentimentAnalysis/
│
├── analyze_sentiment.py          # Cleans text, runs sentiment classification
├── amazon_reviews_raw.csv        # Raw review dataset (10,000 reviews)
├── amazon_reviews_cleaned.csv    # Cleaned dataset with sentiment labels
├── summary.json                  # Aggregated metrics consumed by the dashboard
├── dashboard.html                # Interactive dashboard (open in browser)
└── README.md
```

## Methodology

1. **Data cleaning** — removed duplicates, null/empty reviews, HTML tags, URLs, and non-alphabetic characters from review text.
2. **Sentiment classification** — used **VADER** (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-and-rule-based sentiment analysis tool well-suited for short, informal text like product reviews. Each review got a compound sentiment score, classified as Positive (≥0.05), Negative (≤-0.05), or Neutral (in between).
3. **Validation** — compared predicted sentiment against actual star ratings to measure how well the automated classification matches real customer opinion.
4. **Word frequency analysis** — extracted the most common meaningful words from positive vs negative reviews (after removing stopwords) to surface what customers praise and complain about most.
5. **Trend analysis** — tracked sentiment volume over time by month.
6. **Visualization** — dashboard built with Chart.js: sentiment distribution, sentiment-by-rating breakdown, monthly trend, and word frequency chips.

## Key Insights and Recommendations

| Finding | Recommendation |
|---|---|
| 88.92% of reviews are positive, with a 4.13/5 average rating | Overall product/service quality is strong — leverage this in marketing |
| Model agrees with star ratings 79.16% of the time | NLP sentiment analysis is a reliable way to monitor feedback at scale without reading every review manually |
| "Taste" and "flavor" dominate both positive and negative reviews | Taste consistency is the single biggest driver of customer satisfaction — prioritize quality control here |
| Negative reviews often include "disappointed" and "wouldn't buy again" | These phrases are strong churn-risk signals worth flagging automatically for customer service follow-up |

## How to Run

Just open `dashboard.html` in any browser — all data is embedded, no server or install needed.

To regenerate the analysis from the raw dataset:
```
pip install pandas numpy vaderSentiment
python analyze_sentiment.py
```

## Tools Used

- **Python (pandas, numpy)** — data cleaning and aggregation
- **VADER Sentiment (vaderSentiment)** — rule-based NLP sentiment classification
- **Chart.js** — interactive visualizations (doughnut, stacked bar, line charts)
- **HTML/CSS** — dashboard layout and styling

---

*Submitted as part of Task 4 for the Syntecxhub Data Analysis Internship. Dataset: Amazon Fine Food Reviews (public benchmark dataset).*
