"""File utilities"""

import shutil
import os


def delete_directory(directory_path, corpus_path):
  if os.path.exists(directory_path):
    try:
      shutil.rmtree(directory_path)
    except Exception as e:
      with open(os.path.join(corpus_path, 'error.log'), 'a+') as error_log_file:
        error_log_file.write(f'{e}\n')
  else:
    with open(os.path.join(corpus_path, 'error.log'), 'a+') as error_log_file:
      error_log_file.write(f'no directory {directory_path} to delete\n')
