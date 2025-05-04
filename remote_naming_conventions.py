
import sys
import requests
import re
import yaml
# === Load Configuration ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
PR_NUMBER = str(config["project"]["pr_number"])
OWNER = config["project"]["owner"]
REPO = config["project"]["repo"]
EXTENSIONS = (".cpp", ".h")
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
# === Regex Patterns ===
class_pattern = re.compile(r"\bclass\s+([a-z]\w*)")
var_pattern = re.compile(r"\b(?:int|float|double|char|bool)\s+([A-Z]\w*)")
func_pattern = re.compile(r"\bvoid\s+([A-Z]\w*)\s*\(")
enum_pattern = re.compile(r"\benum\s+([a-z]\w*)")
enum_kind_pattern = re.compile(r"\benum\s+(?!.*Kind\b)(\w+)\b")
# === Exempted Names ===
EXEMPT_NAMES = {'RecursiveASTVisitor'}  # Add any class or function names that you want to exempt
violations = []
current_file = None
line_number = 0
# === Process Diff ===
for line in diff_text.splitlines():
    if line.startswith("+++ b/") and line.endswith(EXTENSIONS):
        current_file = line[6:]
        line_number = 0
        continue
    if not current_file:
        continue
    if line.startswith("@@"):
        match = re.search(r"\+(\d+)", line)
        if match:
            line_number = int(match.group(1)) - 1
        continue
    if line.startswith("+") and not line.startswith("+++"):
        line_number += 1
        code_line = line[1:]
        if (m := class_pattern.search(code_line)):
            class_name = m.group(1)
            # Skip if the class is in the exempted list
            if class_name not in EXEMPT_NAMES:
                violations.append((current_file, line_number, code_line, f"Class '{class_name}' should start with an uppercase letter."))
        if (m := var_pattern.search(code_line)):
            var_name = m.group(1)
            # Skip if the variable is in the exempted list
            if var_name not in EXEMPT_NAMES:
                violations.append((current_file, line_number, code_line, f"Variable '{var_name}' should start with a lowercase letter in camelCase."))
        if (m := func_pattern.search(code_line)):
            func_name = m.group(1)
            # Skip if the function is inimport sys
import requests
import re
import yaml
# === Load Configuration ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
PR_NUMBER = str(config["project"]["pr_number"])
OWNER = config["project"]["owner"]
REPO = config["project"]["repo"]
EXTENSIONS = (".cpp", ".h")
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
# === Regex Patterns ===
class_pattern = re.compile(r"\bclass\s+([a-z]\w*)")
var_pattern = re.compile(r"\b(?:int|float|double|char|bool)\s+([A-Z]\w*)")
func_pattern = re.compile(r"\bvoid\s+([A-Z]\w*)\s*\(")
enum_pattern = re.compile(r"\benum\s+([a-z]\w*)")
enum_kind_pattern = re.compile(r"\benum\s+(?!.*Kind\b)(\w+)\b")
# === Exempted Names ===
EXEMPT_NAMES = {'RecursiveASTVisitor'}  # Add any class or function names that you want to exempt
violations = []
current_file = None
line_number = 0
# === Process Diff ===
for line in diff_text.splitlines():
    if line.startswith("+++ b/") and line.endswith(EXTENSIONS):
        current_file = line[6:]
        line_number = 0
        continue
    if not current_file:
        continue
    if line.startswith("@@"):
        match = re.search(r"\+(\d+)", line)
        if match:
            line_number = int(match.group(1)) - 1
        continue
    if line.startswith("+") and not line.startswith("+++"):
        line_number += 1
        code_line = line[1:]
        if (m := class_pattern.search(code_line)):
            class_name = m.group(1)
            # Skip if the class is in the exempted list
            if class_name not in EXEMPT_NAMES:
                violations.append((current_file, line_number, code_line, f"Class '{class_name}' should start with an uppercase letter."))
        if (m := var_pattern.search(code_line)):
            var_name = m.group(1)
            # Skip if the variable is in the exempted list
            if var_name not in EXEMPT_NAMES:
                violations.append((current_file, line_number, code_line, f"Variable '{var_name}' should start with a lowercase letter in camelCase."))
        if (m := func_pattern.search(code_line)):
            func_name = m.group(1)
            # Skip if the function is in the exempted list
            if func_name not in EXEMPT_NAMES:
                violations.append((current_file, line_number, code_line, f"Function '{func_name}' should start with a lowercase letter in camelCase."))
        if (m := enum_pattern.search(code_line)):
            enum_name = m.group(1)
            violations.append((current_file, line_number, code_line, f"Enum '{enum_name}' should start with an uppercase letter."))
        if (m := enum_kind_pattern.search(code_line)):
            enum_kind_name = m.group(1)
            violations.append((current_file, line_number, code_line, f"Enum type '{enum_kind_name}' should end with 'Kind' if used as a discriminator."))
    elif line.startswith("-") or line.startswith(" "):
        line_number += 1
# === Report Violations ===
if violations:
    print("\n❌ Naming convention violations found:\n")
    for file, line, code, message in violations:
        print(f"🔸 File: {file}, Line: {line}")
        print(f"🔹 Code: {code.strip()}")
        print(f"⚠️  {message}\n")
    sys.exit(1)
else:
    print("\n✅ All modified lines follow naming conventions.")
    sys.exit(0)
