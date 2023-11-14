# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract IR for training.

Extract IR for training, either from a compile_commands.json file produced by
cmake, or a linker parameter list file.

Only run with
'python compiler_opt/tools/extract_ir.py ...'

The compilation is assumed to have been performed with clang, using
-fembed-bitcode=all passed to cc1 (i.e. pass clang -Xclang=-fembed-bitcode=all)

In a distributed ThinLTO case, the compilation is assumed to have been performed
specifying -mllvm -lto-embed-bitcode=post-merge-pre-opt.

In a local ThinLTO case, the compilation is assumedto have been performed
specifying -Wl,--save-temps=import -Wl,--thinlto-emit-index-files

To change the logging verbosity, pass an integer representing the desired
verbosity to the --verbosity flag. Use 0 for all logs, status information,
and detailed debug information, -1 for solely warnings, and -2 to not produce
any output.
"""

import json
import multiprocessing

from absl import app
from absl import flags
from absl import logging

from compiler_opt.tools import extract_ir_lib

flags.DEFINE_string(
    'input', None,
    'Input file or directory - either compile_commands.json, a linker parameter'
    'list, or a path to a directory containing object files.')
flags.DEFINE_enum(
    'input_type', 'json', ['json', 'params', 'directory'],
    'Input file type - json, params, or directory. params latter refers to lld'
    'params.')
flags.DEFINE_string('output_dir', None, 'Output directory')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for objcopy. `None` for maximum available.')
flags.DEFINE_string('llvm_objcopy_path', 'llvm-objcopy', 'Path to llvm-objcopy')
flags.DEFINE_string(
    'obj_base_dir', '',
    'Base directory for object files. Defaults to current working dir.')
flags.DEFINE_string(
    'cmd_filter', None,
    'Include only those modules with a command line matching this regexp. '
    'Setting it to None for not filtering. Note that the regexp is applied '
    'independently for each separate command line option. For example, ^-Oz$ '
    'will match Oz - built binaries. Does not work with thinlto_build=lld.')
flags.DEFINE_enum(
    'thinlto_build', None, ['distributed', 'local'],
    'Set if the build was performed with either \'distributed\' or '
    '\'local\' ThinLTO. This ensures the thinlto.bc files are also copied. '
    'The build is assumed to have had '
    '-mllvm -lto-embed-bitcode=post-merge-pre-opt passed in the distributed '
    'case, or -Wl,--save-temps=import and -Wl,--thinlto-emit-index-files '
    'passed in the local case.')
flags.DEFINE_string(
    'cmd_section_name', '.llvmcmd',
    'The section name passed to llvm-objcopy. For ELF object files, the '
    'default .llvmcmd is correct. For Mach-O object files, one should use '
    'something like __LLVM,__cmdline')
flags.DEFINE_string(
    'bitcode_section_name', '.llvmbc',
    'The section name passed to llvm-objcopy. For ELF object files, the '
    'default .llvmbc is correct. For Mach-O object files, one should use '
    '__LLVM,__bitcode')

flags.mark_flag_as_required('output_dir')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  objs = []
  if FLAGS.input is not None and FLAGS.thinlto_build == 'local':
    raise ValueError('--thinlto_build=local cannot be run with --input')
  if FLAGS.input is None:
    if FLAGS.thinlto_build != 'local':
      raise ValueError('--input or --thinlto_build=local must be provided')
    objs = extract_ir_lib.load_for_lld_thinlto(FLAGS.obj_base_dir,
                                               FLAGS.output_dir)
  elif FLAGS.input_type == 'json':
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = extract_ir_lib.load_from_compile_commands(
          json.load(f), FLAGS.output_dir)
  elif FLAGS.input_type == 'params':
    if not FLAGS.obj_base_dir:
      logging.info(
          '-obj_base_dir is unspecified, assuming current directory.'
          'If no objects are found, use this option to specify the root'
          'directory for the object file paths in the input file.')
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = extract_ir_lib.load_from_lld_params(
          [l.strip() for l in f.readlines()], FLAGS.obj_base_dir,
          FLAGS.output_dir)
  elif FLAGS.input_type == 'directory':
    logging.warning(
        'Using the directory input is only recommended if the build system'
        'your project uses does not support any structured output that'
        'ml-compiler-opt understands. If your build system provides a'
        'structured compilation database, use that instead')
    objs = extract_ir_lib.load_from_directory(FLAGS.input, FLAGS.output_dir)
  else:
    logging.error('Unknown input type: %s', FLAGS.input_type)

  relative_output_paths = extract_ir_lib.run_extraction(
      objs, FLAGS.num_workers, FLAGS.llvm_objcopy_path, FLAGS.cmd_filter,
      FLAGS.thinlto_build, FLAGS.cmd_section_name, FLAGS.bitcode_section_name)

  extract_ir_lib.write_corpus_manifest(FLAGS.thinlto_build,
                                       relative_output_paths, FLAGS.output_dir)

  logging.info('Converted %d files out of %d',
               len(objs) - relative_output_paths.count(None), len(objs))


if __name__ == '__main__':
  multiprocessing.set_start_method('fork')
  app.run(main)
