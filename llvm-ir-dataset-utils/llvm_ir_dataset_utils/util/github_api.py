"""Utilities for accessing parts of the github API"""

import requests


def get_license_from_repo(repo_owner, repo_name, api_token):
  headers = {
      'Accept': 'application/vnd.github+json',
      'Authorization': f'Bearer {api_token}',
      'X-Github-Api-Version': '2022-11-28'
  }
  endpoint = f'https://api.github.com/repos/{repo_owner}/{repo_name}/license'
  # TODO(boomanaiden154): Get rid of verify=False and replace it with a
  # REQUESTS_CA_BUNDLE definition in environments where it is necessary.
  result = requests.get(endpoint, headers=headers, verify=False)
  return result.json()['license']['spdx_id']
