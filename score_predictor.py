import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Small dataset (hours studied -> exam score)
data = {
    "Hours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    "Score": [20, 27, 30, 35, 38, 43, 47, 52, 56, 60, 63, 68, 72, 76, 80, 85, 88, 93],
}
df = pd.DataFrame(data)

# 2. Preprocessing: check for missing values
print(df.isnull().sum())

X = df[["Hours"]]
y = df["Score"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
pred = model.predict(X_test)
print("R2 score:", round(r2_score(y_test, pred), 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 2))


# 6. Take exam name (string only) + hours (int or float) for each exam
def parse_number(text):
    """Convert text to int if possible, otherwise float. Raises ValueError if invalid."""
    try:
        return int(text)
    except ValueError:
        value = float(text)
        if not math.isfinite(value):   # rejects 'nan' and 'inf'
            raise ValueError("not a finite number")
        return value


print("\n--- Score Predictor ---")
print("For each exam, enter the exam name (text only) and the hours studied.")
print("Hours can be whole numbers or decimals: one value = total hours (e.g. 10 or 6.5),")
print("or several values = different days (e.g. 4 5 6 or 4, 5.5, 6).")
print("Press Enter on an empty exam name when you are done.\n")

exam_names = []
totals = []
hours_inputs = []

while True:
    exam_name = input("Exam name (or press Enter to finish): ").strip()
    if exam_name == "":
        break

    # Exam name must be a string of letters/spaces only
    if not exam_name.replace(" ", "").isalpha():
        print("  Exam name must contain letters only (no numbers or symbols). Try again.\n")
        continue

    hours_str = input(f"Hours studied for '{exam_name}': ").strip()
    parts = hours_str.replace(",", " ").split()

    if not parts:
        print("  Please enter at least one number. Try again.\n")
        continue

    try:
        daily_hours = [parse_number(p) for p in parts]
    except ValueError:
        print("  Hours must be numbers only (int or float), like 10, 6.5 or 4 5 6. Try again.\n")
        continue

    if any(h < 0 or h > 24 for h in daily_hours):
        print("  Each day's hours must be between 0 and 24. Try again.\n")
        continue

    total = sum(daily_hours)
    exam_names.append(exam_name.title())
    hours_inputs.append(hours_str)
    totals.append(total)
    print(f"  Added: {exam_name.title()} -> total {total:g} hours\n")

# Predict for all exams at once
predictions = []
if totals:
    predictions = model.predict(pd.DataFrame({"Hours": totals}))
    predictions = [max(0, min(100, p)) for p in predictions]

    result = pd.DataFrame({
        "Exam": exam_names,
        "Hours Input": hours_inputs,
        "Total Hours": totals,
        "Predicted Score": [round(p, 1) for p in predictions],
    })

    print("\n=== Results ===")
    print(result.to_string(index=False))
else:
    print("No exams entered.")

# 7. Visualize (your exams are shown as green points with their names)
plt.scatter(X, y, color="blue", label="Training data")
plt.plot(X, model.predict(X), color="red", label="Regression line")

if totals:
    plt.scatter(totals, predictions, color="green", marker="D", s=70, label="Your exams")
    for name, t, p in zip(exam_names, totals, predictions):
        plt.annotate(name, (t, p), textcoords="offset points", xytext=(5, 8))

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student Score Predictor")
plt.legend()
plt.show()