import subprocess
import requests
import yaml
import sys
import difflib
from termcolor import colored

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)["project"]

def fetch_diff(owner, repo, pr_number):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}.diff"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    print(f"📥 Fetching diff from {url}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed to fetch diff: {resp.status_code}")
        sys.exit(1)
    return resp.text

def run_git_clang_format(base_commit):
    print(f"🎯 Running git clang-format against base {base_commit}...")
    try:
        result = subprocess.run(
            ["git", "clang-format", "--diff", base_commit],
            capture_output=True,
            check=True,
        )
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print("❌ git clang-format failed!")
        print("📤 STDOUT:\n", e.stdout.decode())
        print("📥 STDERR:\n", e.stderr.decode())
        sys.exit(1)

def display_diff_output(diff_output):
    print("\n🧼 Suggested clang-format changes:\n")
    for line in diff_output.splitlines():
        if line.startswith("+"):
            print(colored(line, 'green'))
        elif line.startswith("-"):
            print(colored(line, 'red'))
        else:
            print(line)

def main():
    config = load_config()
    owner = config["owner"]
    repo = config["repo"]
    pr_number = config["pr_number"]

    # Fetch PR diff just for logging
    _ = fetch_diff(owner, repo, pr_number)

    # Assuming PR is already checked out or fetched locally, else you'd need to do that here
    base_commit = config.get("base_commit", "origin/main")

    diff_output = run_git_clang_format(base_commit)

    if "no modified files to format" in diff_output or not diff_output.strip():
        print("✅ No formatting issues found.")
    else:
        display_diff_output(diff_output)

if __name__ == "__main__":
    main()
