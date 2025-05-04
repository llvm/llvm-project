
import subprocess
import requests
import yaml
import sys
 
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
 
def run_clang_format_diff(diff):
    print("🎯 Running clang-format-diff.py on diff...")
    try:
        result = subprocess.run(
            ["./clang-format-diff.py", "-p1", "-binary", "/ptmp/jay/new/llvm-project-jay/build/bin/clang-format"],
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
 
def main():
    config = load_config()
    diff_text = fetch_diff(config["owner"], config["repo"], config["pr_number"])
   
    formatted_output = run_clang_format_diff(diff_text)
 
    if not formatted_output.strip():
        print("✅ No formatting issues found.")
    else:
        print("\n🧼 Suggested clang-format changes:\n")
        print(formatted_output)
 
if __name__ == "__main__":
    main()
