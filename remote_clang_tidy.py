import subprocess
import requests
import yaml
import sys
import os

CONFIG_FILE = "config.yaml"
CLANG_TIDY_DIFF_PATH = "./clang-tidy-diff.py"  # Adjust if stored elsewhere

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def fetch_diff(owner, repo, pr_number):
    print(f"📥 Fetching diff from https://github.com/{owner}/{repo}/pull/{pr_number}.diff")
    url = f"https://github.com/{owner}/{repo}/pull/{pr_number}.diff"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch diff: {response.status_code}")
        sys.exit(1)
    return response.text

def run_clang_tidy_diff(diff_text):
    print("🎯 Running clang-tidy-diff on changed lines...")
    try:
        result = subprocess.run(
    [
        "python3", CLANG_TIDY_DIFF_PATH,
        "-p", "1",
        "-quiet",
        "-j", "4",
        "-clang-tidy-binary", "/ptmp/jay/new/llvm-project-checks/build/bin/clang-tidy"
    ],
    input=diff_text.encode("utf-8"),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
        print(result.stdout.decode("utf-8"))
        if result.stderr:
            print("⚠️ Errors/Warnings:", result.stderr.decode("utf-8"), file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to run clang-tidy-diff: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(CLANG_TIDY_DIFF_PATH):
        print(f"❌ clang-tidy-diff.py not found at {CLANG_TIDY_DIFF_PATH}")
        sys.exit(1)

    config = load_config()
    project = config.get("project", {})
    owner = project.get("owner")
    repo = project.get("repo")
    pr_number = project.get("pr_number")

    if not all([owner, repo, pr_number]):
        print("❌ Missing configuration: owner, repo, or pr_number")
        sys.exit(1)

    diff_text = fetch_diff(owner, repo, pr_number)
    run_clang_tidy_diff(diff_text)

if __name__ == "__main__":
    main()
