
import pandas as pd
import random
from datetime import date, timedelta

random.seed(42)

locations = [
    "Sector 17",
    "Airport Road",
    "Zirakpur Highway",
    "Kharar Road",
    "Phase 7",
    "Landran Road",
    "Industrial Area",
    "Sector 22"
]

severities = ["Minor", "Moderate", "Severe", "Fatal"]
weather_types = ["Clear", "Rainy", "Foggy", "Cloudy"]
road_conditions = ["Dry", "Wet", "Poor", "Under Repair"]
times = ["Morning", "Afternoon", "Evening", "Night"]

records = []
start_date = date(2026, 1, 1)

for i in range(100):
    severity = random.choices(
        severities,
        weights=[45, 30, 20, 5]
    )[0]

    records.append({
        "accident_id": f"DEMO{i + 1:03d}",
        "date": (
            start_date
            + timedelta(days=random.randint(0, 270))
        ).isoformat(),
        "location": random.choice(locations),
        "severity": severity,
        "weather": random.choice(weather_types),
        "road_condition": random.choice(road_conditions),
        "time_of_day": random.choice(times),
        "vehicles_involved": random.randint(1, 4),
        "fatalities": (
            1 if severity == "Fatal"
            and random.random() < 0.7 else 0
        ),
        "injuries": random.randint(0, 5),
        "source_type": "SYNTHETIC_DEMO"
    })

df = pd.DataFrame(records)

df.to_csv("sample_accidents.csv", index=False)

print("Sample CSV file successfully created!")
print("Total demo records:", len(df))
print(df.head())
