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
    print(f"🎯 Running git-clang-format against base {base_commit}...")
    try:
        result = subprocess.run(
            [
                "python3",
                "/ptmp/jay/new/llvm-project-checks/clang/tools/clang-format/git-clang-format",
                "--diff",
                base_commit,
                "--binary",
                "/ptmp/jay/new/llvm-project-checks/build/bin/clang-format"  # replace with your actual path
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print("❌ git-clang-format failed!")
        print("📤 STDOUT:\n", e.stdout.decode())
        print("📥 STDERR:\n", e.stderr.decode())
        sys.exit(1)

def display_diff_output(diff_output):
    print("\n🧼 Suggested clang-format changes:\n")
    
    # Display the diff in a user-friendly format
    diff = difflib.unified_diff(
        diff_output.splitlines(), 
        lineterm='', 
        fromfile="Original File", 
        tofile="Formatted File"
    )
    
    # This part helps display the context more clearly
    for line in diff:
        if line.startswith("+"):
            print(colored(line, 'green'))  # Suggested change
        elif line.startswith("-"):
            print(colored(line, 'red'))  # The original line that is being removed/changed
        else:
            print(line)  # Context lines

def main():
    config = load_config()
    owner = config["owner"]
    repo = config["repo"]
    pr_number = config["pr_number"]
    base_commit = config.get("base_commit", "origin/main")

    _ = fetch_diff(owner, repo, pr_number)  # Just logging, not used further

    diff_output = run_git_clang_format(base_commit)

    if "no modified files to format" in diff_output or not diff_output.strip():
        print("✅ No formatting issues found.")
    else:
        print("📝 Here are the suggested formatting changes (lines marked with '+' are added, '-' are removed):")
        display_diff_output(diff_output)

if __name__ == "__main__":
    main()
