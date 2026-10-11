import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path="spam.csv"):
    """Load the SMS spam dataset and prepare it for training."""
    df = pd.read_csv(path, encoding="latin-1", engine="python")

    # Typical Kaggle/UCI spam.csv columns are v1 = label and v2 = message
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]
    else:
        # Fallback for a cleaned version with columns label and message
        if {"label", "message"}.issubset(df.columns):
            df = df[["label", "message"]]
        else:
            raise ValueError(
                "Dataset must contain either columns ('v1', 'v2') or ('label', 'message')."
            )

    df = df.drop_duplicates()
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def train_model(df):
    """Train the SMS spam classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    model = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)),
        MultinomialNB(),
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Label distribution in dataset:")
    print(df["label"].value_counts())
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))

    return model


def check_message(model, text):
    """Predict whether a message is spam or ham."""
    label = model.predict([text])[0]
    prob = model.predict_proba([text])[0][1]
    category = "SPAM" if label else "HAM"
    return f"{category} (spam probability: {prob:.2f})"


def main():
    df = load_data("spam.csv")
    model = train_model(df)

    print("\nExample predictions:")
    print(check_message(model, "Congratulations! You won a free iPhone. Click here to claim now!"))
    print(check_message(model, "Hey, are we still meeting for lunch tomorrow?"))


if __name__ == "__main__":
    main()
