
import subprocess
import sys
import re
import requests
import yaml
# === Load config.yaml ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# === Configuration ===
PR_NUMBER = str(config["project"]["pr_number"])
OWNER = config["project"]["owner"]
REPO = config["project"]["repo"]
headers = {
    "Accept": "application/vnd.github.v3.diff"
}
# === Fetch PR Diff ===
url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}"
diff_url = f"{url}.diff"
print(f"📥 Fetching diff from {diff_url}")
resp = requests.get(diff_url, headers=headers)
if resp.status_code != 200:
    print(f"❌ Failed to fetch PR diff: {resp.status_code} {resp.text}")
    sys.exit(1)
diff_text = resp.text
if not diff_text.strip():
    print("✅ No changes in the PR.")
    sys.exit(0)
# === Run clang-tidy-diff.py on diff from stdin ===
print("🧼 Running clang-tidy-diff.py on PR diff...")
# Adjust if clang-tidy-diff.py is in a different path
clang_tidy_diff_path = "clang-tidy-diff.py"
result = subprocess.run(
    ["python3", clang_tidy_diff_path, "-p1"],
    input=diff_text,
    text=True,
    capture_output=True
)
# === Output Results ===
if result.returncode == 0 and not result.stdout.strip():
    print("✅ No clang-tidy issues detected!")
    sys.exit(0)
else:
    print("🚨 Issues detected:")
    print("\n================= Diff Before clang-tidy =================")
    print(diff_text)
    print("\n================= Suggested Fixes =================")
    print(result.stdout)
    if result.stderr:
        print("\n⚠️ Error while running clang-tidy:")
        print(result.stderr)
    sys.exit(1)
