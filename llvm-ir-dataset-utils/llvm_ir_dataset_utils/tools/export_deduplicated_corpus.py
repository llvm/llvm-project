"""Tool for taking in a list of module hashes and extracting all deduplicated
modules into a separate directory."""

import os
import logging
import csv
import shutil
import pathlib
import json
import random

from absl import flags
from absl import app

import ray

from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import parallel
from llvm_ir_dataset_utils.util import licenses

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('module_hash_list', None,
                          'A list of module hashes to pull from.')
flags.DEFINE_string(
    'output_path', None,
    'The output path to place all the deduplicated modules into.')
flags.DEFINE_integer('batch_size', 256,
                     'The number of modules to put in each batch.')
flags.DEFINE_boolean(
    'split_by_corpora', True,
    'Whether or not to put separate corpora (defined by module hash lists) into separate folders.'
)
flags.DEFINE_multi_string(
    'project_license_info', [],
    'A JSON file containing license information on a set of projects.'
    'Setting this flag will force the script to validate license information')

flags.mark_flag_as_required('module_hash_list')
flags.mark_flag_as_required('output_path')


def load_module_hashes(file_path):
  logging.info(f'Loading data from {file_path}')
  module_hash_map = {}
  corpus_name = os.path.splitext(os.path.basename(file_path))[0]
  all_modules_count = 0
  with open(file_path) as module_hashes_file:
    module_hash_reader = csv.DictReader(module_hashes_file)
    for module_hash_entry in module_hash_reader:
      all_modules_count += 1
      module_hash = module_hash_entry['module_hashes']
      file_path = module_hash_entry['name']
      # Skip empty modules which get hashes to the default value of 4
      if module_hash == '4':
        continue
      module_hash_map[module_hash] = (file_path, corpus_name)
  logging.info(f'Read {all_modules_count} modules.')
  logging.info(f'Found {len(module_hash_map)} unique modules.')
  return module_hash_map


def load_project_licenses(file_path):
  logging.info(f'Loading license data from {file_path}')
  project_license_map = {}
  with open(file_path) as project_licenses_file:
    project_license_array = json.load(project_licenses_file)
    for project_license_info in project_license_array:
      corpus_name = project_license_info[0]
      license_id = project_license_info[1]
      license_source = project_license_info[2]
      license_files = project_license_info[3]
      license_file_ids = [
          license_file["license"] for license_file in license_files
      ]
      source_url = project_license_info[4]
      if licenses.is_license_valid(license_id, license_file_ids):
        project_license_map[corpus_name] = {
            'license_id': license_id,
            'license_source': license_source,
            'license_files': license_files,
            'source_url': source_url
        }
  logging.info(
      f'Finished loading license info from {file_path}, found {len(project_license_map)} valid licenses'
  )
  return project_license_map


def create_manifest(folder_path, modules_list):
  file_path = os.path.join(folder_path, 'corpus_description.json')
  corpus_description = {'has_thinlto': False, 'modules': []}
  for module_tuple in modules_list:
    # Omit the .bc file extension because it gets added on by different
    # tooling.
    corpus_description['modules'].append(f'{module_tuple[1]}')
  with open(file_path, 'w') as corpus_description_file:
    json.dump(corpus_description, corpus_description_file, indent=2)

  license_info_file_path = os.path.join(folder_path, 'license_info.json')
  license_info_map = {}
  for module_tuple in modules_list:
    module_file_path, module_hash, corpus_name, license_info = module_tuple
    license_info_map[module_hash] = license_info
  with open(license_info_file_path, 'w') as license_info_file:
    json.dump(license_info_map, license_info_file, indent=2)


@ray.remote(num_cpus=1)
def process_module_batch(batch_path, modules_to_process):
  pathlib.Path(batch_path).mkdir(parents=True)
  for module_path in modules_to_process:
    file_path_full = module_path[0]
    module_hash = module_path[1]
    file_path_parts = file_path_full.split(':')
    bitcode_file = dataset_corpus.load_file_from_corpus(file_path_parts[0],
                                                        file_path_parts[1])
    with open(os.path.join(batch_path, f'{module_hash}.bc'),
              'wb') as bitcode_file_handle:
      bitcode_file_handle.write(bitcode_file)

    # Process the .cmd file
    command_line_file_path = file_path_parts[1][:-3] + '.cmd'
    command_line_data = ''
    if dataset_corpus.is_file_in_corpus(file_path_parts[0],
                                        command_line_file_path):
      command_line_data = dataset_corpus.load_file_from_corpus(
          file_path_parts[0], command_line_file_path).decode('utf-8')
    with open(os.path.join(batch_path, f'{module_hash}.cmd'),
              'w') as command_line_file_handle:
      command_line_file_handle.write(command_line_data)

    # Process .source and .preprocessed_source files
    source_path = file_path_parts[1][:-3] + '.source'
    source_data = ''
    if dataset_corpus.is_file_in_corpus(file_path_parts[0], source_path):
      source_data = dataset_corpus.load_file_from_corpus(
          file_path_parts[0], source_path).decode('utf-8')
    with open(os.path.join(batch_path, f'{module_hash}.source'),
              'w') as source_file_handle:
      source_file_handle.write(source_data)

    preprocessed_source_path = file_path_parts[1][:-3] + '.preprocessed_source'
    preprocessed_source_data = ''
    if dataset_corpus.is_file_in_corpus(file_path_parts[0],
                                        preprocessed_source_path):
      preprocessed_source_data = dataset_corpus.load_file_from_corpus(
          file_path_parts[0], preprocessed_source_path).decode('utf-8')
    with open(
        os.path.join(batch_path, f'{module_hash}.preprocessed_source'),
        'w') as preprocessed_source_file_handle:
      preprocessed_source_file_handle.write(preprocessed_source_data)

  create_manifest(batch_path, modules_to_process)
  shutil.make_archive(batch_path, 'tar', batch_path)
  shutil.rmtree(batch_path)


def extract_files_from_hash_map(module_hash_map, output_path):
  modules_to_process = {}

  for module_hash in module_hash_map:
    # Each key in the map accesses a tuple with the format(file path, corpus name)
    # format to use is (path, hash)
    file_path, corpus_name, license_id, license_source, license_files, source_url = module_hash_map[
        module_hash]
    tuple_to_append = (file_path, module_hash, (license_id, license_source,
                                                license_files, source_url))
    if corpus_name in modules_to_process:
      modules_to_process[corpus_name].append(tuple_to_append)
    else:
      modules_to_process[corpus_name] = [tuple_to_append]

  for corpus_name in modules_to_process:
    random.shuffle(modules_to_process[corpus_name])

  module_batches = []

  for corpus_name in modules_to_process:
    current_module_batches = parallel.split_batches(
        modules_to_process[corpus_name], FLAGS.batch_size)
    output_module_batches = []
    for current_module_batch in current_module_batches:
      # Once the issue in the parallel module related to creating empty batches
      # on boundaries gets fixed, remove this.
      # TODO(boomanaiden154)
      if len(current_module_batch) == 0:
        continue
      output_module_batch = []
      for module_info in current_module_batch:
        file_path, module_hash, license_info = module_info
        output_module_batch.append(
            (file_path, module_hash, corpus_name, license_info))
      output_module_batches.append(output_module_batch)
    module_batches.extend(output_module_batches)

  module_batch_futures = []

  for index, module_batch in enumerate(module_batches):
    if FLAGS.split_by_corpora:
      corpus_name = module_batch[0][2]
      batch_path = os.path.join(FLAGS.output_path, corpus_name,
                                f'batch-{index}')
    else:
      batch_path = os.path.join(FLAGS.output_path, f'batch-{index}')
    module_batch_futures.append(
        process_module_batch.remote(batch_path, module_batch))

  while len(module_batch_futures) > 0:
    finished, module_batch_futures = ray.wait(module_batch_futures, timeout=5.0)
    finished_data = ray.get(finished)
    logging.info(
        f'Just finished {len(finished_data)}, {len(module_batch_futures)} remaining.'
    )


def check_and_add_module_licenses(module_hash_map, license_info_map):
  # We're adding a tuple onto the end of every module_hash, particularly
  # with the values (license_id, license_source, license_files)
  validated_module_hash_map = {}
  if len(license_info_map) == 0:
    for module_hash in module_hash_map:
      validated_module_hash_map[module_hash] = module_hash_map[module_hash] + (
          None, None, None, None)
  logging.info('Checking module licenses')
  for module_hash in module_hash_map:
    file_path, corpus_name = module_hash_map[module_hash]
    corpus_archive_path = file_path.split(':')[0]
    if corpus_archive_path in license_info_map:
      extra_license_info = (
          license_info_map[corpus_archive_path]['license_id'],
          license_info_map[corpus_archive_path]['license_source'], [
              license_struct['file'] for license_struct in
              license_info_map[corpus_archive_path]['license_files']
          ], license_info_map[corpus_archive_path]['source_url'])
      validated_module_hash_map[
          module_hash] = module_hash_map[module_hash] + extra_license_info
  logging.info(
      f'Finished checking module licenses, ended up with {len(validated_module_hash_map)}'
      f'out of {len(module_hash_map)} original modules.')
  return validated_module_hash_map


def main(_):
  ray.init()

  pathlib.Path(FLAGS.output_path).mkdir(exist_ok=True, parents=True)

  module_hash_map = {}

  for module_hash_list_path in FLAGS.module_hash_list:
    module_hash_map.update(load_module_hashes(module_hash_list_path))

  license_info_map = {}

  for project_license_list_path in FLAGS.project_license_info:
    license_info_map.update(load_project_licenses(project_license_list_path))

  module_hash_map = check_and_add_module_licenses(module_hash_map,
                                                  license_info_map)

  extract_files_from_hash_map(module_hash_map, FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)
