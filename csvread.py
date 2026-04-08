import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# PLAYER CLASS
# -----------------------------
class Player:
    def __init__(self, name, team, role, strike_rate, average, wickets, economy):
        self.name = name
        self.team = team
        self.role = role
        self.strike_rate = strike_rate
        self.average = average
        self.wickets = wickets
        self.economy = economy

        # Match stats
        self.runs = 0
        self.balls = 0
        self.fours = 0
        self.sixes = 0
        self.out = False


# -----------------------------
# LOAD DATA
# -----------------------------
def load_players():
    data = pd.read_csv("player_name-WPS Office.csv")

    # Clean column names (IMPORTANT FIX)
    data.columns = data.columns.str.strip()

    players = []
    for _, row in data.iterrows():
        players.append(Player(
            row['player_name'],
            row['team'],
            row['role'],
            row['strike_rate'],
            row['average'],
            row['bowling_strike_rate'],
            row['economy']
        ))

    return players


# -----------------------------
# MATCH PHASE
# -----------------------------
def get_phase(over):
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle"
    else:
        return "death"


# -----------------------------
# BASIC LOGIC (for training)
# -----------------------------
def ball_outcome_basic(batsman, bowler, phase):
    prob = random.random()

    if prob < 0.05:
        return "W"
    elif prob < 0.30:
        return 0
    elif prob < 0.55:
        return 1
    elif prob < 0.75:
        return 2
    elif prob < 0.90:
        return 4
    else:
        return 6


# -----------------------------
# GENERATE TRAINING DATA
# -----------------------------
def generate_data(players, samples=5000):
    dataset = []

    for _ in range(samples):
        batsman = random.choice(players)
        bowler = random.choice(players)
        phase = random.choice(["powerplay", "middle", "death"])

        result = ball_outcome_basic(batsman, bowler, phase)

        dataset.append([
            batsman.strike_rate,
            batsman.average,
            bowler.economy,
            bowler.wickets,
            phase,
            result
        ])

    return dataset


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(players):
    dataset = generate_data(players)

    df = pd.DataFrame(dataset, columns=[
        "sr", "avg", "eco", "wkts", "phase", "result"
    ])

    df['phase'] = df['phase'].map({
        "powerplay": 0,
        "middle": 1,
        "death": 2
    })

    df['result'] = df['result'].replace("W", -1)

    # Weight wickets higher
    df['weight'] = df['result'].apply(lambda x: 5 if x == -1 else 1)

    X = df[["sr", "avg", "eco", "wkts", "phase"]]
    y = df["result"]
    w = df["weight"]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train, sample_weight=w_train)

    print("Model trained!")

    return model


# -----------------------------
# ML PREDICTION (FIXED)
# -----------------------------
def ball_outcome_ml(batsman, bowler, phase, model):
    phase_map = {"powerplay": 0, "middle": 1, "death": 2}

    features = pd.DataFrame([{
        "sr": batsman.strike_rate,
        "avg": batsman.average,
        "eco": bowler.economy,
        "wkts": bowler.wickets,
        "phase": phase_map[phase]
    }])

    probs = model.predict_proba(features)[0]
    classes = model.classes_

    pred = random.choices(classes, weights=probs)[0]

    if pred == -1:
        return "W"
    return int(pred)


# -----------------------------
# SIMULATE INNINGS (FIXED)
# -----------------------------
def simulate_innings(batting_team, bowling_team, model):
    runs = 0
    wickets = 0
    striker = 0
    non_striker = 1
    bowler_index = 0

    for over in range(20):
        phase = get_phase(over)
        print(f"\nOver {over+1} ({phase})")

        bowler = bowling_team[bowler_index % len(bowling_team)]

        for ball in range(6):
            if wickets == 10:
                return runs

            batsman = batting_team[striker]
            result = ball_outcome_ml(batsman, bowler, phase, model)

            batsman.balls += 1

            if result == "W":
                print(f"{batsman.name} OUT!")
                batsman.out = True
                wickets += 1
                striker = wickets + 1
                if striker >= len(batting_team):
                    return runs
            else:
                batsman.runs += result

                if result == 4:
                    batsman.fours += 1
                elif result == 6:
                    batsman.sixes += 1

                runs += result
                print(f"{batsman.name} scored {result}")

                if result % 2 == 1:
                    striker, non_striker = non_striker, striker

        striker, non_striker = non_striker, striker
        bowler_index += 1

    return runs


# -----------------------------
# SCORECARD
# -----------------------------
def print_scorecard(team, name):
    print(f"\n🏏 Scorecard - {name}")
    print("-" * 45)
    print("Name\tRuns\tBalls\t4s\t6s\tSR")

    for p in team:
        sr = (p.runs / p.balls * 100) if p.balls > 0 else 0
        status = "" if p.out else "*"
        print(f"{p.name}{status}\t{p.runs}\t{p.balls}\t{p.fours}\t{p.sixes}\t{sr:.1f}")


# -----------------------------
# MATCH SIMULATION
# -----------------------------
def simulate_match(players, model):
    teams = list(set([p.team for p in players]))

    team1 = [p for p in players if p.team == teams[0]]
    team2 = [p for p in players if p.team == teams[1]]

    print(f"\nMatch: {teams[0]} vs {teams[1]}")

    print(f"\n{teams[0]} Batting:")
    score1 = simulate_innings(team1, team2, model)
    print_scorecard(team1, teams[0])

    print(f"\nTotal: {score1}")
    print(f"\nTarget: {score1 + 1}")

    print(f"\n{teams[1]} Batting:")
    score2 = simulate_innings(team2, team1, model)
    print_scorecard(team2, teams[1])

    print(f"\nTotal: {score2}")

    if score1 > score2:
        print(f"\n{teams[0]} WON!")
    elif score2 > score1:
        print(f"\n{teams[1]} WON!")
    else:
        print("\nMATCH DRAW!")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    players = load_players()
    model = train_model(players)
    simulate_match(players, model)