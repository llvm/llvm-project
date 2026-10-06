"""Module for building and extracting bitcode from Julia applications"""

import subprocess
import os
import pathlib
import json
import logging
import shutil
import glob

from mlgo.corpus import make_corpus_lib
"""
Generates the command to compile a bitcode archive from a Julia package.
The archive then needs to be unpacked with `ar -x`.
"""


def generate_build_command(package_to_build, thread_count):
  command_vector = [
      "julia",
      "--threads",
      f"{thread_count}",
      "--quiet",
  ]

  # Close out the Julia command line switches
  command_vector.append("--")

  julia_builder_jl_path = os.path.join(
      os.path.dirname(__file__), 'julia_builder.jl')
  command_vector.append(julia_builder_jl_path)

  # Add the package to build
  command_vector.append(package_to_build)

  return command_vector


def perform_build(package_name, build_dir, corpus_dir, thread_count):
  build_command_vector = generate_build_command(package_name, thread_count)

  build_log_name = f'./{package_name}.build.log'
  build_log_path = os.path.join(corpus_dir, build_log_name)

  environment = os.environ.copy()
  julia_depot_path = os.path.join(build_dir, 'julia_depot')
  environment['JULIA_DEPOT_PATH'] = julia_depot_path
  environment['JULIA_PKG_SERVER'] = ''
  julia_bc_path = os.path.join(build_dir, 'unopt_bc')
  os.mkdir(julia_bc_path)
  environment['JULIA_PKG_UNOPT_BITCODE_DIR'] = julia_bc_path
  environment['JULIA_IMAGE_THREADS'] = '1'
  environment['JULIA_CPU_TARGET'] = 'x86-64'

  try:
    with open(build_log_path, 'w') as build_log_file:
      subprocess.run(
          build_command_vector,
          cwd=build_dir,
          stdout=build_log_file,
          stderr=build_log_file,
          env=environment,
          check=True)
  except subprocess.SubprocessError:
    logging.warn(f'Failed to build julia package {package_name}')
    build_success = False
  else:
    build_success = True
  if build_success:
    extract_ir(build_dir, corpus_dir)
  return {
      'targets': [{
          'success': build_success,
          'build_log': build_log_name,
          'name': package_name
      }]
  }


def unpack_archives(unopt_bc_archive_dir, unopt_bc_dir):
  archive_files = os.listdir(unopt_bc_archive_dir)
  for archive_file in archive_files:
    full_archive_file_path = os.path.join(unopt_bc_archive_dir, archive_file)
    # Strip the last two characters which will be the .a in the extensions
    archive_package_name = archive_file[:-2]

    archive_extraction_command_vector = ['llvm-ar', '-x', archive_file]

    subprocess.run(
        archive_extraction_command_vector,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=unopt_bc_archive_dir)

    # Copy text_opt#0.bc to the output directory
    unopt_bitcode_full_path = os.path.join(unopt_bc_archive_dir,
                                           'text_unopt#0.bc')
    copied_bitcode_full_path = os.path.join(unopt_bc_dir,
                                            f'{archive_package_name}.bc')
    shutil.copyfile(unopt_bitcode_full_path, copied_bitcode_full_path)

    # Delete all bitcode files from the current extraction in preparation
    # for the next archive.
    for bitcode_file in glob.glob(os.path.join(unopt_bc_archive_dir, '*.bc')):
      os.remove(bitcode_file)

    os.remove(full_archive_file_path)


def extract_ir(build_dir, corpus_dir):
  unopt_bc_dir = os.path.join(build_dir, 'unopt_bc')
  output_bc_dir = os.path.join(build_dir, 'output_bc')
  os.mkdir(output_bc_dir)
  unpack_archives(unopt_bc_dir, output_bc_dir)
  relative_paths = make_corpus_lib.load_bitcode_from_directory(output_bc_dir)
  make_corpus_lib.copy_bitcode(relative_paths, output_bc_dir, corpus_dir)
  make_corpus_lib.write_corpus_manifest(relative_paths, corpus_dir, '')
