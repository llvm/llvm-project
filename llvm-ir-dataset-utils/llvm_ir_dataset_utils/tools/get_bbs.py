"""Tool for extracting basic blocks from the corpus"""

import os
import logging
import subprocess
import json
import binascii
import tempfile

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import parallel

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('corpus_dir', None,
                          'The corpus directory to look for project in.')
flags.DEFINE_string('output_file', None,
                    'The output file to put unique BBs in.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('output_file')

OPT_PASS_LIST = ['default<O0>', 'default<O1>', 'default<O2>', 'default<O3>']
LLC_OPT_LEVELS = ['-O0', '-O1', '-O2', '-O3']

PROJECT_MODULE_CHUNK_SIZE = 8


def get_basic_blocks(input_file_path, module_id):
  command_vector = ['extract_bbs_from_obj', input_file_path]
  command_output = subprocess.run(
      command_vector, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if (command_output.returncode != 0):
    logging.warning(f'Failed to get basic blocks from {module_id}')
    return []
  return command_output.stdout.decode('utf-8').split('\n')


def output_optimized_bc(input_file_path, pass_list, output_file_path):
  opt_command_vector = [
      'opt', f'-passes={pass_list}', input_file_path, '-o', output_file_path
  ]
  opt_output = subprocess.run(
      opt_command_vector, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  return opt_output.returncode == 0


def get_asm_lowering(input_file_path, opt_level, output_file_path, module_id):
  llc_command_vector = [
      'llc', opt_level, input_file_path, '-filetype=obj',
      '-basic-block-sections=labels', '-o', output_file_path
  ]
  llc_output = subprocess.run(
      llc_command_vector, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if llc_output.returncode != 0:
    logging.warning(f'Failed to lower {module_id} to asm')
  return llc_output.returncode == 0


def process_bitcode_file(bitcode_file_path, module_id):
  # Get basic blocks from the unoptimized version, and at all optimization levels.
  basic_blocks = []
  with tempfile.TemporaryDirectory() as temp_dir:
    for index, opt_pass in enumerate(OPT_PASS_LIST):
      bc_output_path = os.path.join(temp_dir, f'{index}.bc')
      opt_return = output_optimized_bc(bitcode_file_path, opt_pass,
                                       bc_output_path)
      # If we run into an error (output_optimized_bc returns false), continue onto
      # the next iteration and log a warning.
      if not opt_return:
        logging.warning(f'Failed to optimized {module_id}')
        continue
      for index, llc_level in enumerate(LLC_OPT_LEVELS):
        asm_output_path = f'{bc_output_path}.{index}.o'
        asm_lowering_output = get_asm_lowering(bc_output_path, llc_level,
                                               asm_output_path, module_id)
        # Only get the basic blocks from the lowered file if we successfully
        # lower the bitcode.
        if asm_lowering_output:
          basic_blocks.extend(get_basic_blocks(asm_output_path, module_id))

  return list(set(basic_blocks))


@ray.remote(num_cpus=1)
def process_modules_batch(modules_batch):
  basic_blocks = []

  for bitcode_module_info in modules_batch:
    project_path, bitcode_module = bitcode_module_info
    module_data = dataset_corpus.load_file_from_corpus(project_path,
                                                       bitcode_module)
    module_id = f'{project_path}:{bitcode_module}'
    if module_data is None:
      continue

    with tempfile.NamedTemporaryFile() as temp_bc_file:
      temp_bc_file.write(module_data)
      basic_blocks.extend(
          process_bitcode_file(temp_bc_file.file.name, module_id))

  return list(set(basic_blocks))


# TODO(boomanaiden154): Abstract the infrastructure to parse modules into
# batches into somewhere common as it is used in several places already,
# including grep_source.py.
@ray.remote(num_cpus=1)
def get_bc_files_in_project(project_path):
  try:
    bitcode_modules = dataset_corpus.get_bitcode_file_paths(project_path)
  except Exception:
    return []

  return [(project_path, bitcode_module) for bitcode_module in bitcode_modules]


def get_bbs_from_projects(project_list, output_file_path):
  logging.info(f'Processing {len(project_list)} projects.')

  project_info_futures = []

  for project_path in project_list:
    project_info_futures.append(get_bc_files_in_project.remote(project_path))

  project_infos = []

  while len(project_info_futures) > 0:
    to_return = 32 if len(project_info_futures) > 64 else 1
    finished, project_info_futures = ray.wait(
        project_info_futures, timeout=5.0, num_returns=to_return)
    logging.info(
        f'Just finished gathering modules from {len(finished)} projects, {len(project_info_futures)} remaining.'
    )
    for finished_project in ray.get(finished):
      project_infos.extend(finished_project)

  logging.info(
      f'Finished gathering modules, currently have {len(project_infos)}')

  module_batches = parallel.split_batches(project_infos,
                                          PROJECT_MODULE_CHUNK_SIZE)

  logging.info(f'Setup {len(module_batches)} batches.')

  module_batch_futures = []

  for module_batch in module_batches:
    module_batch_futures.append(process_modules_batch.remote(module_batch))

  with open(output_file_path, 'w') as output_file_handle:
    while len(module_batch_futures) > 0:
      to_return = 32 if len(module_batch_futures) > 64 else 1
      finished, module_batch_futures = ray.wait(
          module_batch_futures, timeout=5.0, num_returns=to_return)
      logging.info(
          f'Just finished {len(finished)} batches, {len(module_batch_futures)} remaining.'
      )
      for finished_batch in ray.get(finished):
        for basic_block in finished_batch:
          output_file_handle.write(f'{basic_block}\n')


def main(_):
  project_dirs = []

  for corpus_dir in FLAGS.corpus_dir:
    for project_dir in os.listdir(corpus_dir):
      project_dirs.append(os.path.join(corpus_dir, project_dir))

  get_bbs_from_projects(project_dirs, FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
