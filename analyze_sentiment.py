import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- Step 1: Load ----
df = pd.read_csv("amazon_reviews_raw.csv")
print("Raw rows:", len(df))
print("Nulls:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# ---- Step 2: Clean ----
df_clean = df.drop_duplicates()
df_clean = df_clean.dropna(subset=["Text", "Score"])
df_clean = df_clean[df_clean["Text"].str.len() > 10]  # remove near-empty reviews

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML tags
    text = re.sub(r'http\S+', ' ', text)         # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)        # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_clean["CleanText"] = df_clean["Text"].apply(clean_text)
df_clean = df_clean[df_clean["CleanText"].str.len() > 5]

print("\nAfter cleaning:", len(df_clean), "rows")

# ---- Step 3: Sentiment classification (VADER) ----
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

results = df_clean["CleanText"].apply(get_sentiment)
df_clean["Sentiment"] = results.apply(lambda x: x[0])
df_clean["SentimentScore"] = results.apply(lambda x: x[1])

df_clean.to_csv("amazon_reviews_cleaned.csv", index=False)

# ---- Step 4: Distribution & patterns ----
sentiment_counts = df_clean["Sentiment"].value_counts().to_dict()
sentiment_pct = (df_clean["Sentiment"].value_counts(normalize=True) * 100).round(2).to_dict()

# Sentiment vs star rating (validates our model against real user ratings)
rating_sentiment = df_clean.groupby("Score")["Sentiment"].value_counts(normalize=True).unstack().fillna(0).round(3) * 100

# Accuracy check: does Score 4-5 mostly align with Positive sentiment, 1-2 with Negative?
df_clean["RatingLabel"] = df_clean["Score"].apply(lambda s: "Positive" if s >= 4 else ("Negative" if s <= 2 else "Neutral"))
agreement = (df_clean["Sentiment"] == df_clean["RatingLabel"]).mean() * 100

# ---- Step 5: Common words per sentiment ----
stopwords = set("""the a an and or but is are was were be been being this that these those
i me my we our you your he him his she her it its they them their to of in on at for with
as by from up down out so not no very just too than then also have has had do does did
product br""".split())

def top_words(texts, n=12):
    words = []
    for t in texts:
        words.extend([w for w in t.split() if w not in stopwords and len(w) > 2])
    return Counter(words).most_common(n)

pos_words = top_words(df_clean[df_clean["Sentiment"]=="Positive"]["CleanText"])
neg_words = top_words(df_clean[df_clean["Sentiment"]=="Negative"]["CleanText"])

# ---- Step 6: Trend over time ----
df_clean["Date"] = pd.to_datetime(df_clean["Time"], unit="s")
df_clean["YearMonth"] = df_clean["Date"].dt.to_period("M").astype(str)
monthly_sentiment = df_clean.groupby("YearMonth")["Sentiment"].value_counts().unstack().fillna(0)
monthly_sentiment = monthly_sentiment.sort_index()
# keep a reasonable number of points for chart
if len(monthly_sentiment) > 24:
    monthly_sentiment = monthly_sentiment.iloc[-24:]

summary = {
    "total_reviews": int(len(df_clean)),
    "sentiment_counts": {k: int(v) for k, v in sentiment_counts.items()},
    "sentiment_pct": sentiment_pct,
    "avg_rating": round(df_clean["Score"].mean(), 2),
    "model_agreement_pct": round(agreement, 2),
    "rating_sentiment_breakdown": rating_sentiment.round(1).to_dict(),
    "top_positive_words": pos_words,
    "top_negative_words": neg_words,
    "monthly_trend": {
        col: monthly_sentiment[col].to_dict() for col in monthly_sentiment.columns
    }
}

with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== RESULTS ===")
print("Sentiment counts:", sentiment_counts)
print("Sentiment %:", sentiment_pct)
print("Avg star rating:", summary["avg_rating"])
print("Model agreement with star ratings:", round(agreement,2), "%")
print("\nTop positive words:", pos_words[:8])
print("Top negative words:", neg_words[:8])
