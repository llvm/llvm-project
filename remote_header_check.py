
import sys
import subprocess
import re
import yaml
import requests
# === Load config.yaml ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# === Configuration ===
PR_NUMBER = str(config["project"]["pr_number"])
OWNER = config["project"]["owner"]
REPO = config["project"]["repo"]
LLVM_HEADER_TEMPLATE = "//===----------------------------------------------------------------------===//"
LLVM_LICENSE = "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception"
FILE_EXTENSIONS = (".cpp", ".h")
# === GitHub PR Diff URL ===
DIFF_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}.diff"
# === Fetch PR Diff ===
print(f"📥 Fetching PR diff for PR #{PR_NUMBER}...")
response = requests.get(DIFF_URL, headers={"Accept": "application/vnd.github.v3.diff"})
if response.status_code != 200:
    print(f"❌ Failed to fetch PR diff. Status: {response.status_code}")
    sys.exit(1)
diff_text = response.text
if not diff_text.strip():
    print("✅ No changes in the PR.")
    sys.exit(0)
# === Extract Modified .cpp/.h Files ===
pr_files = [
    line.split(" ")[1][2:] for line in diff_text.splitlines()
    if line.startswith("+++ b/") and line.endswith(FILE_EXTENSIONS)
]
if not pr_files:
    print("✅ No .cpp or .h files modified in this PR.")
    sys.exit(0)
print("\n🔍 Checking headers in the following modified files:")
for file in pr_files:
    print("  •", file)
# === Check Each File for LLVM Header in Modified Lines ===
missing_header_files = []
for file in pr_files:
    raw_url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/pull/{PR_NUMBER}/head/{file}"
    file_response = requests.get(raw_url)
    if file_response.status_code != 200:
        continue  # Skip without printing a message if file is not found in PR head
    content = file_response.text
    # Extract modified lines from the diff
    modified_lines = [
        line[1:] for line in diff_text.splitlines() if line.startswith("+") and line[1:] not in content
    ]
    
    # Check only the modified lines for the header
    header_found = any(
        LLVM_HEADER_TEMPLATE in line or LLVM_LICENSE in line for line in modified_lines
    )
    if not header_found:
        print(f"\n❌ Missing or incorrect LLVM-style header in the modified lines of: {file}")
        print("Expected header must include:")
        print(f"  {LLVM_HEADER_TEMPLATE}")
        print(f"  {LLVM_LICENSE}")
        missing_header_files.append(file)
# === Final Report ===
if missing_header_files:
    print(f"\n❌ {len(missing_header_files)} file(s) missing proper LLVM-style headers in modified lines.")
    sys.exit(1)
else:
    print("\n✅ All modified files contain correct LLVM-style headers in modified lines!")
    sys.exit(0)
