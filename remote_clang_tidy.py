import requests
import subprocess
import json
import os

# Function to load configuration details from the JSON file
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

# Function to fetch diff data from GitHub using the API
def fetch_diff_from_github(owner, repo, pull_number, token):
    """Fetch the diff for the pull request using GitHub API."""
    url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files'
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Extracting the diff for each file changed in the pull request
        diff_data = []
        for file in response.json():
            if 'patch' in file:
                diff_data.append(file['patch'])
        return "\n".join(diff_data)
    else:
        raise Exception(f"Failed to fetch diff: {response.status_code} {response.text}")

# Function to run clang-tidy-diff.py on the provided diff data
def run_clang_tidy_diff(diff_data, clang_tidy_path="./clang-tidy-diff.py"):
    """Run clang-tidy-diff on the provided diff data."""
    # Create a temporary file to hold the diff content
    with open("temp_diff.patch", "w") as diff_file:
        diff_file.write(diff_data)
    
    # Run the clang-tidy-diff script with the diff file
    command = ['python3', clang_tidy_path]
    with open("temp_diff.patch", "r") as diff_file:
        process = subprocess.Popen(command, stdin=diff_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print("Error:", stderr.decode())
        else:
            print("clang-tidy output:\n", stdout.decode())

# Main function to integrate everything
def main(config_file="config.json"):
    # Step 1: Load configuration from the config file
    try:
        config = load_config(config_file)
        owner = config['owner']
        repo = config['repo']
        pull_number = config['pull_number']
        github_token = config['github_token']
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Step 2: Fetch the diff data from GitHub API
    try:
        diff_data = fetch_diff_from_github(owner, repo, pull_number, github_token)
        if diff_data:
            # Step 3: Pass the diff to clang-tidy-diff.py script
            run_clang_tidy_diff(diff_data)
        else:
            print("No changes found in the pull request.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main(config_file="config.json")
