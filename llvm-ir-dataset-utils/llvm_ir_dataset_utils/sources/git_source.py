"""Module that downloads git repositories"""

import os
import subprocess
import logging


def download_source_code(repo_url, repo_name, commit_sha, base_dir, corpus_dir):
  # If the directory already exists, we can skip downloading the source,
  # currently just assuming that the requested commit is present
  if not os.path.exists(os.path.join(base_dir, repo_name)):
    with open(os.path.join(corpus_dir, 'git.log'), 'w') as git_log_file:
      git_command_vector = ["git", "clone", repo_url]
      if commit_sha is None or commit_sha == '':
        git_command_vector.append('--depth=1')
      git_command_vector.append(repo_name)
      logging.info(f"Cloning git repository {repo_url}")
      environment = os.environ.copy()
      environment['GIT_TERMINAL_PROMPT'] = '0'
      environment['GIT_ASKPASS'] = 'echo'
      try:
        subprocess.run(
            git_command_vector,
            cwd=base_dir,
            stdout=git_log_file,
            stderr=git_log_file,
            env=environment,
            check=True)
        if commit_sha is not None and commit_sha != '':
          commit_checkout_vector = ["git", "checkout", commit_sha]
          logging.info(f"Checked out commit SHA {commit_sha}")
          subprocess.run(
              commit_checkout_vector,
              cwd=os.path.join(base_dir, repo_name),
              stdout=git_log_file,
              stderr=git_log_file,
              check=True)
        success = True
      except subprocess.SubprocessError:
        logging.warning(
            f'Cloning and checking out git repository {repo_url} failed.')
        success = False
  else:
    success = True
  return {
      'type': 'git',
      'repo_url': repo_url,
      'commit_sha': commit_sha,
      'success': success
  }
