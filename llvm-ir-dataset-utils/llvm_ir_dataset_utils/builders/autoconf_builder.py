"""Module for building and extracting bitcode from applications using autoconf"""

import os
import subprocess

from mlgo.corpus import extract_ir_lib

CONFIGURE_LOG_NAME = './configure.log'
BUILD_LOG_NAME = './build.log'


def generate_configure_command(root_path, options_dict):
  command_vector = [os.path.join(root_path, "configure")]
  for option in options_dict:
    command_vector.append(f"--{option}=\"{options_dict[option]}\"")
  return command_vector


def generate_build_command(threads):
  command_vector = ["make", f"-j{threads}"]
  return command_vector


def perform_build(configure_command_vector, build_command_vector, build_dir,
                  corpus_dir):
  configure_env = os.environ.copy()
  configure_env["CC"] = "clang"
  configure_env["CXX"] = "clang++"
  configure_env["CFLAGS"] = "-Xclang -fembed-bitcode=all"
  configure_env["CXXFLAGS"] = "-Xclang -fembed-bitcode=all"
  configure_command = " ".join(configure_command_vector)
  configure_log_path = os.path.join(corpus_dir, CONFIGURE_LOG_NAME)
  with open(configure_log_path, 'w') as configure_log_file:
    configure_process = subprocess.run(
        configure_command,
        cwd=build_dir,
        env=configure_env,
        shell=True,
        stdout=configure_log_file,
        stderr=configure_log_file)
    configure_success = configure_process.returncode == 0
  build_log_path = os.path.join(corpus_dir, BUILD_LOG_NAME)
  with open(build_log_path, 'w') as build_log_file:
    build_process = subprocess.run(
        build_command_vector,
        cwd=build_dir,
        stdout=build_log_file,
        stderr=build_log_file)
    build_success = build_process.returncode == 0
  return {
      'targets': [{
          'success': build_success and configure_success,
          'build_log': BUILD_LOG_NAME,
          'configure_log': CONFIGURE_LOG_NAME,
          'name': os.path.basename(corpus_dir),
          'build_success': build_success,
          'configure_success': configure_success
      }]
  }


def extract_ir(build_dir, corpus_dir, threads):
  objects = extract_ir_lib.load_from_directory(build_dir, corpus_dir)
  relative_output_paths = extract_ir_lib.run_extraction(objects, threads,
                                                        "llvm-objcopy", None,
                                                        None, ".llvmcmd",
                                                        ".llvmbc")
  extract_ir_lib.write_corpus_manifest(None, relative_output_paths, corpus_dir)
