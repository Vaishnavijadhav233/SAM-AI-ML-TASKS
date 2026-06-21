import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- Sample Dataset ---------------- #

texts = [
    "Congratulations you won a lottery",
    "Claim your free reward now",
    "Win cash instantly click here",
    "Limited offer hurry up",
    "Your parcel has arrived",
    "Let's go for lunch",
    "Meeting at 10 AM tomorrow",
    "Happy Birthday have a nice day",
    "Can you call me later",
    "Project submission is today"
]

labels = [
    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Ham",
    "Ham",
    "Ham",
    "Ham",
    "Ham",
    "Ham"
]

# ---------------- Train Model ---------------- #

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# ---------------- Prediction Function ---------------- #

def detect():
    msg = entry.get()

    if msg == "":
        messagebox.showwarning("Warning", "Enter SMS First!")
        return

    test = vectorizer.transform([msg])
    result = model.predict(test)[0]

    if result == "Spam":
        output.config(
            text="🚫 SPAM MESSAGE",
            fg="red"
        )
    else:
        output.config(
            text="✅ NOT SPAM",
            fg="green"
        )

# ---------------- GUI ---------------- #

root = tk.Tk()
root.title("AI Spam SMS Detector")
root.geometry("700x500")
root.configure(bg="#74b9ff")

title = tk.Label(
    root,
    text="📱 AI SPAM SMS DETECTOR",
    font=("Arial", 24, "bold"),
    bg="#74b9ff",
    fg="navy"
)

title.pack(pady=20)

tk.Label(
    root,
    text="Enter SMS:",
    font=("Arial", 15),
    bg="#74b9ff"
).pack()

entry = tk.Entry(
    root,
    font=("Arial", 15),
    width=45
)

entry.pack(pady=15)

button = tk.Button(
    root,
    text="🔍 Detect",
    font=("Arial", 15, "bold"),
    bg="orange",
    fg="white",
    command=detect
)

button.pack(pady=15)

output = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 20, "bold"),
    bg="#74b9ff"
)

output.pack(pady=30)

footer = tk.Label(
    root,
    text="Machine Learning Internship Project",
    font=("Arial", 11),
    bg="#74b9ff",
    fg="purple"
)

footer.pack(side="bottom", pady=15)

root.mainloop()
