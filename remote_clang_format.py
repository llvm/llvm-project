import subprocess
import requests
import yaml
import sys
import difflib
from termcolor import colored  # Import termcolor to use colored output

# Function to load configuration from the YAML file
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)["project"]

# Function to fetch the diff from the GitHub API
def fetch_diff(owner, repo, pr_number):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}.diff"
    headers = {"Accept": "application/vnd.github.v3.diff"}
    print(f"📥 Fetching diff from {url}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed to fetch diff: {resp.status_code}")
        sys.exit(1)
    return resp.text

# Function to run the clang-format-diff.py and get the formatted output
def run_clang_format_diff(diff):
    print("🎯 Running clang-format-diff.py on diff...")
    try:
        result = subprocess.run(
            ["python3", "./clang-format-diff.py", "-p1", "-binary", "/ptmp/jay/new/llvm-project-checks/build/bin/clang-format"],
            input=diff.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print("❌ clang-format-diff.py failed!")
        print("📤 STDOUT:")
        print(e.stdout.decode())
        print("📥 STDERR:")
        print(e.stderr.decode())
        sys.exit(1)

# Function to display before and after code with green coloring for the corrected code
def display_changes(before, after):
    """
    This function will compare the before and after code and display the difference to the user.
    The actual corrected code will be printed in green.
    """
    print("\n===================================")
    print("            BEFORE FORMATTING      ")
    print("===================================")
    print(before)
    print("\n===================================")
    print("            AFTER FORMATTING      ")
    print("===================================")
    
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='')
    print("\n🧼 Suggested clang-format changes:\n")
    
    for line in diff:
        if line.startswith("+"):  # Lines that were added or corrected
            print(colored(line, 'green'))  # Print the corrected lines in green
        else:
            print(line)

# Main function to load configuration, fetch diff, and run clang-format-diff.py
def main():
    config = load_config()
    diff_text = fetch_diff(config["owner"], config["repo"], config["pr_number"])

    # Simulate before and after formatting
    formatted_output = run_clang_format_diff(diff_text)

    if not formatted_output.strip():
        print("✅ No formatting issues found.")
    else:
        # Get the "before" code from the diff (those with + or - sign)
        before_code = ""
        after_code = ""

        # Extract the diff lines and display before/after comparison
        for line in diff_text.splitlines():
            if line.startswith('-'):
                before_code += line[1:] + "\n"  # Before formatting (lines with '-')
            elif line.startswith('+'):
                after_code += line[1:] + "\n"  # After formatting (lines with '+')

        # Now display the changes with green color for the corrected code
        display_changes(before_code, formatted_output)

if __name__ == "__main__":
    main()
