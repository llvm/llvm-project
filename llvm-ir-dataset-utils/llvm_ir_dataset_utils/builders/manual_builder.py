"""Module for building and extracting bitcode from applications using an
arbitrary build system by manually running specified commands."""

import subprocess
import os

from mlgo.corpus import extract_ir_lib
from mlgo.corpus import make_corpus_lib

BUILD_LOG_NAME = './build.log'


def perform_build(commands_list, build_dir, threads, corpus_dir,
                  environment_variables):
  command_statuses = []
  build_log_path = os.path.join(corpus_dir, BUILD_LOG_NAME)
  for command in commands_list:
    environment = os.environ.copy()
    environment['JOBS'] = str(threads)
    for environment_variable in environment_variables:
      environment[environment_variable] = environment_variables[
          environment_variable]
    with open(build_log_path, 'w') as build_log_file:
      build_process = subprocess.run(
          command,
          cwd=build_dir,
          env=environment,
          shell=True,
          stderr=build_log_file,
          stdout=build_log_file)
      command_statuses.append(build_process.returncode == 0)
  overall_success = True
  for command_status in command_statuses:
    if not command_status:
      overall_success = False
      break
  return {
      'targets': [{
          'success': overall_success,
          'build_log': BUILD_LOG_NAME,
          'name': os.path.basename(corpus_dir)
      }]
  }


def extract_ir(build_dir, corpus_dir, threads):
  objects = extract_ir_lib.load_from_directory(build_dir, corpus_dir)
  relative_output_paths = extract_ir_lib.run_extraction(objects, threads,
                                                        "llvm-objcopy", None,
                                                        None, ".llvmcmd",
                                                        ".llvmbc")
  extract_ir_lib.write_corpus_manifest(None, relative_output_paths, corpus_dir)


def extract_raw_ir(build_dir, corpus_dir, threads):
  relative_paths = make_corpus_lib.load_bitcode_from_directory(build_dir)
  make_corpus_lib.copy_bitcode(relative_paths, build_dir, corpus_dir)
  make_corpus_lib.write_corpus_manifest(relative_paths, corpus_dir, '')
