#!/bin/python
"""This script wraps the compiler, taking in the compiler options and saving
the source files that are used within the compilation step."""

import os
import subprocess
import sys
import shutil

RECOGNIZED_SOURCE_FILE_EXTENSIONS = ['.c', '.cpp', '.cxx', '.cc']


def run_compiler_invocation(mode, compiler_arguments):
  command_vector = []

  if mode == 'c++':
    command_vector.append('clang++')
  else:
    command_vector.append('clang')

  command_vector.extend(compiler_arguments)

  compiler_process = subprocess.run(command_vector)

  return compiler_process.returncode


def save_preprocessed_source(mode, compiler_arguments):
  # We shouldn't fail to find the output here if the argument parsing
  # succeeded.
  output_index = compiler_arguments.index('-o') + 1
  arguments_copy = compiler_arguments.copy()
  output_path = arguments_copy[output_index] + '.preprocessed_source'
  arguments_copy[output_index] = output_path

  # Add -E to the compiler invocation to run just the preprocessor.
  arguments_copy.append('-E')

  run_compiler_invocation(mode, arguments_copy)


def save_source(source_files, output_file, mode, compiler_arguments):
  assert (len(source_files) <= 1)
  for source_file in source_files:
    new_file_name = output_file + '.source'
    shutil.copy(source_file, new_file_name)

    save_preprocessed_source(mode, compiler_arguments)


def parse_args(arguments_split):
  mode = 'c++'
  if not arguments_split[0].endswith('++'):
    mode = 'c'

  output_file_path = None
  try:
    output_arg_index = arguments_split.index('-o') + 1
    output_file_path = arguments_split[output_arg_index]
  except Exception:
    return (mode,)

  input_files = []

  for argument in arguments_split:
    for recognized_extension in RECOGNIZED_SOURCE_FILE_EXTENSIONS:
      if argument.endswith(recognized_extension):
        input_files.append(argument)

  return (output_file_path, input_files, mode)


def main(args):
  parsed_arguments = parse_args(args)
  if len(parsed_arguments) == 1:
    # We couldn't parse the arguments. This could be for a varietey of reasons.
    # In this case, don't copy over any files and just run the compiler
    # invocation.
    mode = parsed_arguments
    return_code = run_compiler_invocation(mode, args[1:])
    sys.exit(return_code)

  output_file_path, input_files, mode = parsed_arguments

  save_source(input_files, output_file_path, mode, args[1:])

  return_code = run_compiler_invocation(mode, args[1:])
  sys.exit(return_code)


if __name__ == '__main__':
  main(sys.argv)
