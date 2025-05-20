import subprocess
import requests
import yaml
import tempfile
import os
import sys

# === Config and Constants ===
CONFIG_FILE = "config.yaml"
CLANG_TIDY_DIFF = "clang-tidy-diff.py"
CLANG_TIDY_BINARY = "./build/bin/clang-tidy"  # Adjust if needed

# === Load PR info from config.yaml ===
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

PR_NUMBER = str(config["project"]["pr_number"])
OWNER = config["project"]["owner"]
REPO = config["project"]["repo"]

# === Fetch PR diff ===
print(f"📥 Fetching diff from PR #{PR_NUMBER}")
diff_url = f"https://github.com/{OWNER}/{REPO}/pull/{PR_NUMBER}.diff"
response = requests.get(diff_url)
if response.status_code != 200:
    print(f"❌ Failed to fetch PR diff: {response.status_code}")
    sys.exit(1)

diff_text = response.text
if not diff_text.strip():
    print("✅ No changes detected in the PR.")
    sys.exit(0)

# === Save to temporary file for piping
with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.diff') as tmp:
    tmp.write(diff_text)
    tmp_path = tmp.name

# === Run clang-tidy-diff.py ===
print(f"🎯 Running clang-tidy-diff on PR #{PR_NUMBER}...")
try:
    result = subprocess.run(
        [
            "python3", CLANG_TIDY_DIFF,
            "-p", "1",
            "-quiet",
            "-clang-tidy-binary", CLANG_TIDY_BINARY,
            "-checks", "*",
            "-export-fixes", "temp_fixes.yaml"
        ],
        input=diff_text.encode("utf-8"),
        capture_output=True,
        check=False
    )

    stdout = result.stdout.decode("utf-8")
    stderr = result.stderr.decode("utf-8")

    print(stdout)
    if stderr:
        print("⚠️ Errors/Warnings:\n", stderr)

    if "warning:" not in stdout and "error:" not in stdout:
        print("✅ No clang-tidy issues found.")

except Exception as e:
    print(f"❌ Failed to run clang-tidy-diff: {e}")
    sys.exit(1)
finally:
    os.remove(tmp_path)

