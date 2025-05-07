# remote_clang_tidy_suggestions.py

import subprocess
import requests
import yaml
import sys
import os
import tempfile

CONFIG_FILE = "config.yaml"
CLANG_TIDY_DIFF_PATH = "./clang-tidy-diff.py"
CLANG_TIDY_BINARY = "/ptmp/jay/new/llvm-project-checks/build/bin/clang-tidy"
SUGGESTION_DB_FILE = "suggestions.yaml"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def load_suggestion_db():
    if not os.path.exists(SUGGESTION_DB_FILE):
        return {}
    with open(SUGGESTION_DB_FILE, "r") as f:
        return yaml.safe_load(f)

def fetch_diff(owner, repo, pr_number):
    print(f"\U0001F4E5 Fetching diff from https://github.com/{owner}/{repo}/pull/{pr_number}.diff")
    url = f"https://github.com/{owner}/{repo}/pull/{pr_number}.diff"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"\u274C Failed to fetch diff: {response.status_code}")
        sys.exit(1)
    return response.text

def parse_yaml_diagnostics(fix_file, suggestion_map):
    if not os.path.exists(fix_file):
        return []
    with open(fix_file, "r") as f:
        data = yaml.safe_load(f)
    
    diagnostics = data.get("Diagnostics", [])
    suggestions = []
    for diag in diagnostics:
        name = diag.get("DiagnosticName")
        message = diag.get("Message")
        path = diag.get("FilePath")
        line = diag.get("FileOffset", "?")
        
        suggestions.append(f"\n\U0001F4CD In {path}, line offset {line}: {message}")
        for fix in suggestion_map.get(name, ["No specific suggestion. Please review manually."]):
            suggestions.append(f"   \U0001F527 {fix}")
    return suggestions

def run_clang_tidy_diff(diff_text, suggestion_map):
    print("\U0001F3AF Running clang-tidy-diff on changed lines...")
    try:
        tmp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml").name

        result = subprocess.run(
            [
                "python3", CLANG_TIDY_DIFF_PATH,
                "-p", "1",
                "-quiet",
                "-j", "4",
                "-clang-tidy-binary", CLANG_TIDY_BINARY,
                "-checks", "-*,-clang-analyzer-*,-cppcoreguidelines-*,-modernize-*,-readability-*,-llvm-*,-bugprone-*,-performance-*,-misc-*,-google-*,-hicpp-*",
                "-export-fixes", tmp_yaml
            ],
            input=diff_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        print(result.stdout.decode("utf-8"))
        stderr_text = result.stderr.decode("utf-8")
        if stderr_text:
            print("\u26A0\uFE0F Errors/Warnings:\n", stderr_text, file=sys.stderr)

        suggestions = parse_yaml_diagnostics(tmp_yaml, suggestion_map)
        if suggestions:
            print("\n\U0001F6E0 Suggested Human Fixes:")
            print("\n".join(suggestions))
        else:
            print("\n✅ No actionable clang-tidy suggestions found.")

    except Exception as e:
        print(f"\u274C Failed to run clang-tidy-diff: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(CLANG_TIDY_DIFF_PATH):
        print(f"\u274C clang-tidy-diff.py not found at {CLANG_TIDY_DIFF_PATH}")
        sys.exit(1)

    config = load_config()
    suggestion_map = load_suggestion_db()
    project = config.get("project", {})
    owner = project.get("owner")
    repo = project.get("repo")
    pr_number = project.get("pr_number")

    if not all([owner, repo, pr_number]):
        print("\u274C Missing configuration: owner, repo, or pr_number")
        sys.exit(1)

    diff_text = fetch_diff(owner, repo, pr_number)
    run_clang_tidy_diff(diff_text, suggestion_map)

if __name__ == "__main__":
    main()
