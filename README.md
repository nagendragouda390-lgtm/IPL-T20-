# IPL-T20-
🏏 AI-Powered T20 Cricket Match Simulator

A machine learning-based cricket match simulator that predicts ball-by-ball outcomes and generates realistic scorecards for T20 matches.

---

🚀 Features

- 🎯 Ball-by-ball simulation using ML model
- 🤖 Random Forest based prediction engine
- 📊 Real-time scorecard with:
  - Runs, balls, 4s, 6s
  - Strike rate
  - Not-out tracking
- 🔄 Dynamic match flow (powerplay, middle, death overs)
- ⚖️ Probabilistic predictions (not fixed outputs)

---

🧠 How It Works

1. Player data is loaded from CSV
2. Synthetic match data is generated
3. A Random Forest model is trained
4. Each ball outcome is predicted using probability
5. Match is simulated with real-time updates

---

🛠️ Tech Stack

- Python
- pandas
- scikit-learn
- Random Forest Classifier

---

📂 Project Structure

project/
│── main.py
│── player_name-WPS Office.csv
│── README.md

---

▶️ How to Run

1. Install dependencies:

pip install pandas scikit-learn

2. Run the project:

python main.py

---

📊 Sample Output

Match: MI vs DC

Over 1 (powerplay)
Rohit Sharma scored 4
Rohit Sharma scored 1
Ishan Kishan scored 2
...

🏏 Scorecard - MI
---------------------------------------------
Name        Runs  Balls  4s  6s  SR


Rohit Sharma*   45   30   5   2   150.0

---

⚠️ Limitations

- Uses synthetic training data (not real match data)
- Predictions are probabilistic, not exact
- No UI (CLI-based output)

---

🔥 Future Improvements

- Use real IPL ball-by-ball dataset
- Build web app using Streamlit
- Add player-vs-player statistics
- Visual dashboards (graphs, charts)

---

💡 Learning Outcomes

- Machine Learning pipeline
- Feature engineering basics
- Handling imbalanced data
- Simulation system design
- Debugging real-world issues

---

👨‍💻 Author

Developed as part of AIML learning journey.

---

⭐ If you like this project

Give it a star and share feedback!
