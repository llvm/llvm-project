"""Module that automatically downloads source code based on a source
description."""

from llvm_ir_dataset_utils.sources import git_source
from llvm_ir_dataset_utils.sources import tar_source


def download_source(source_descriptions, base_dir, corpus_dir, folder_name):
  source_logs = []
  for source_description in source_descriptions:
    if (source_description['type'] == 'git'):
      source_logs.append(
          git_source.download_source_code(source_description['repo_url'],
                                          folder_name,
                                          source_description['commit_sha'],
                                          base_dir, corpus_dir))
    elif (source_description['type'] == 'tar'):
      source_logs.append(
          tar_source.download_source_code(source_description['archive_url'],
                                          base_dir, folder_name))
    if source_logs[-1]['success']:
      return source_logs
  return source_logs
