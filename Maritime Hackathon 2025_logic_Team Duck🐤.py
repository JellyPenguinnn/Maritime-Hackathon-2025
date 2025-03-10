import pandas as pd
import re
import os

# Step 1: Load the Dataset
try:
    train_data = pd.read_csv("psc_severity_train.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the file 'psc_severity_train.csv' is in the project directory.")
    exit()

# Step 2: Define Majority Voting (4 possible labels)
def derive_consensus(severities):
    """
    Derive consensus severity using majority voting and priority-based fallback,
    now with a 4th label: 'Not a deficiency'.
    If there's a clear majority, pick that label. If not, fallback to:
      1) High
      2) Medium
      3) Low
      4) Not a deficiency
    in that order.
    """
    # Tally the labels
    severity_counts = severities.value_counts()
    
    # If there's a single label that appears most frequently, return it
    if severity_counts.max() > 1:
        return severity_counts.idxmax()
    
    # Otherwise, there's no majority (all labels appear once)
    # We'll check in priority order: 'High' > 'Medium' > 'Low' > 'Not a deficiency'
    if "High" in severities.values:
        return "High"
    elif "Medium" in severities.values:
        return "Medium"
    elif "Low" in severities.values:
        return "Low"
    elif "Not a deficiency" in severities.values:
        return "Not a deficiency"
    else:
        # Fallback if something unexpected
        return "Low"

# Step 3: Parse `def_text` into Structured Fields
def parse_deficiency_text(def_text):
    """Parse the structured deficiency text into individual fields."""
    fields = {
        "Immediate Causes": "",
        "Root Cause Analysis": "",
        "Corrective Action": "",
        "Preventive Action": "",
    }
    if pd.isna(def_text):  # Handle missing or empty `def_text`
        return fields
    for field in fields.keys():
        match = re.search(f"{field}:(.*?)(?=\n|$)", def_text, re.DOTALL)
        if match:
            fields[field] = match.group(1).strip()
    return fields

# Step 4: Compute Text-Based Severity (unchanged)
def compute_text_severity(parsed_fields):
    """Compute severity based on parsed fields and dynamically adjusted weights."""
    base_weights = {
        "Immediate Causes": 0.3,
        "Root Cause Analysis": 0.25,
        "Corrective Action": 0.2,
        "Preventive Action": 0.1,
    }
    available_fields = {field: weight for field, weight in base_weights.items() if parsed_fields[field]}
    total_weight = sum(available_fields.values())
    if total_weight == 0:
        return "Medium"  # Default severity if all fields are empty
    normalized_weights = {field: weight / total_weight for field, weight in available_fields.items()}
    severity_score = sum(
        2 * weight if "critical" in parsed_fields[field].lower()
        else weight if "urgent" in parsed_fields[field].lower()
        else 0
        for field, weight in normalized_weights.items()
    )
    if severity_score >= 0.5:
        return "High"
    elif 0.3 <= severity_score < 0.5:
        return "Medium"
    else:
        return "Low"

# Step 5: Derive Consensus Severity
try:
    consensus_severity = (
        train_data.groupby(["PscInspectionId", "deficiency_code"])["annotation_severity"]
        .apply(derive_consensus)
        .reset_index(name="consensus_severity")
    )
    print("Consensus severity derived successfully.")
except KeyError as e:
    print(f"Error: {e}. Ensure the dataset has the required columns.")
    exit()

# Step 6: Parse `def_text` and Compute Text-Based Severity
train_data["parsed_fields"] = train_data["def_text"].apply(parse_deficiency_text)
train_data["text_based_severity"] = train_data["parsed_fields"].apply(compute_text_severity)

# Step 7: Merge Consensus Severity with Original Data
try:
    important_columns = [
        "PscInspectionId", "deficiency_code", "annotation_id", "annotation_severity",
        "def_text", "VesselGroup", "age", "InspectionDate", "PscAuthorityId", "PortId"
    ]
    unique_data = train_data.drop_duplicates(subset=["PscInspectionId", "deficiency_code"])[important_columns]
    final_data = unique_data.merge(consensus_severity, on=["PscInspectionId", "deficiency_code"], how="left")
    final_data["final_consensus_severity"] = final_data["consensus_severity"]
    print("Final dataset created successfully.")
except KeyError as e:
    print(f"Error: {e}. Ensure the dataset has the required columns.")
    exit()

# Step 8: Save the Final Dataset
output_file = "final_consensus_severity.csv"
try:
    final_data.to_csv(output_file, index=False)
    print(f"Final dataset saved to '{output_file}'.")
    if os.path.exists(output_file):
        print(f"File '{output_file}' successfully created in the project directory.")
    else:
        print(f"File '{output_file}' not found in directory.")
except Exception as e:
    print(f"Error saving the file: {e}")
    exit()
