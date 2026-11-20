"""A script for uploading a dataset in the form of a folder of parquet files
to huggingface.
"""

import logging
import os

from absl import app
from absl import flags

import ray

from huggingface_hub import HfApi
from huggingface_hub import CommitOperationAdd
from huggingface_hub import preupload_lfs_files
from huggingface_hub import create_commit

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_dir', None,
                    'The path to the folder containing the parquet files.')
flags.DEFINE_string('commit_message', None,
                    'Git commit message for the upload.')
flags.DEFINE_string('start_after', None, 'A specific path to start at.')
flags.DEFINE_integer('operations_per_commit', 50,
                     'The number of operations to cache before committing')

flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('commit_message')


@ray.remote(num_cpus=4)
def upload_file(api, full_file_path, file_to_upload):
  try:
    hf_file_path = 'data/' + file_to_upload
    operation = CommitOperationAdd(
        path_in_repo=hf_file_path, path_or_fileobj=full_file_path)
    preupload_lfs_files(
        'llvm-ml/ComPile', additions=[operation], repo_type='dataset')
    logging.warning(f'Finished uploading {file_to_upload}')
    return (True, operation)
  except Exception as e:
    logging.error(f'Ran into an error, retrying {file_to_upload}: {e}')
    return (False, full_file_path, file_to_upload)


def main(_):
  logging.info('Starting the upload')
  api = HfApi()

  file_upload_futures = []

  for language_folder in os.listdir(FLAGS.dataset_dir):
    for file_name in os.listdir(
        os.path.join(FLAGS.dataset_dir, language_folder)):
      if FLAGS.start_after and file_to_upload <= FLAGS.start_after:
        logging.info(f'Skipping uploading {file_to_upload}')
        continue

      full_file_path = os.path.join(FLAGS.dataset_dir, language_folder,
                                    file_name)

      file_to_upload = os.path.join(language_folder, file_name)
      file_upload_futures.append(
          upload_file.remote(api, full_file_path, file_to_upload))

  current_operations = []

  while len(file_upload_futures) > 0:
    completed_uploads, file_upload_futures = ray.wait(
        file_upload_futures, timeout=5)

    logging.info(
        f'Just finished {len(completed_uploads)}, {len(file_upload_futures)} remaining.'
    )
    returned_uploads = ray.get(completed_uploads)

    for returned_upload in returned_uploads:
      if returned_upload[0]:
        current_operations.append(returned_upload[1])
      else:
        file_upload_futures.append(
            upload_file.remote(api, returned_upload[0], returned_upload[1]))

    if len(current_operations) > FLAGS.operations_per_commit:
      create_commit(
          'llvm-ml/ComPile',
          operations=current_operations,
          commit_message='Add additional data',
          repo_type='dataset')
      current_operations = []

  create_commit(
      'llvm-ml/ComPile',
      operations=current_operations,
      commit_message=FLAGS.commit_message,
      repo_type='dataset')
  current_operations = []


if __name__ == '__main__':
  app.run(main)
