"""Module for building and extracting bitcode from Swift packages."""

import subprocess
import os
import logging

from mlgo.corpus import extract_ir_lib

BUILD_TIMEOUT = 900

BUILD_LOG_NAME = './build.log'


def perform_build(source_dir, build_dir, corpus_dir, thread_count,
                  package_name):
  build_command_vector = [
      'swift', 'build', '-c', 'release', '-Xswiftc', '-embed-bitcode',
      '--emit-swift-module-separately', '-Xswiftc', '-Onone', '-j',
      str(thread_count), '--build-path', build_dir
  ]

  build_log_path = os.path.join(corpus_dir, BUILD_LOG_NAME)

  try:
    with open(build_log_path, 'w') as build_log_file:
      subprocess.run(
          build_command_vector,
          cwd=source_dir,
          stdout=build_log_file,
          stderr=build_log_file,
          check=True,
          timeout=BUILD_TIMEOUT)
  except (subprocess.SubprocessError, FileNotFoundError):
    # TODO(boomanaiden154): Figure out why a FileNotFoundError is thrown here
    # sometimes because it should be handled earlier.
    logging.warning(f'Failed to build swift package in {package_name}')
    build_success = False
  else:
    build_success = True
  if build_success:
    extract_ir(build_dir, corpus_dir, thread_count)
  return {
      'targets': [{
          'success': build_success,
          'build_log': BUILD_LOG_NAME,
          'name': package_name
      }]
  }


def extract_ir(build_dir, corpus_dir, threads):
  objects = extract_ir_lib.load_from_directory(build_dir, corpus_dir)
  relative_output_paths = extract_ir_lib.run_extraction(
      objects, threads, "llvm-objcopy", None, None, "__LLVM,__swift_cmdline",
      "__LLVM,__bitcode")
  extract_ir_lib.write_corpus_manifest(None, relative_output_paths, corpus_dir)
