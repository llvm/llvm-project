"""Utilities for working with bitcode modules."""

import subprocess
import os
import tempfile
import logging
import json
import shutil
import textwrap

import ray

from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import pass_list_constants
from llvm_ir_dataset_utils.util import parallel

BITCODE_FILE_CHUNK_SIZE = 16

OPT_TIMEOUT_SECONDS = 60
FASTBPE_TIMEOUT_SECONDS = 180
LLVM_DIS_TIMEOUT_SECONDS = 180


def get_function_symbols(bitcode_module):
  llvm_nm_command_vector = ['llvm-nm', '--defined-only', '--format=posix', '-']
  with subprocess.Popen(
      llvm_nm_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      stdin=subprocess.PIPE) as llvm_nm_process:
    stdout = llvm_nm_process.communicate(
        input=bitcode_module)[0].decode('utf-8')
    if llvm_nm_process.returncode != 0:
      logging.warning('Failed to get functions from bitcode module.')
      return (stdout.replace('\n', ''), None)
    module_symbols = stdout.split('\n')[:-1]
  module_list = []
  for symbol in module_symbols:
    symbol_parts = symbol.split(' ')
    # Only look for t or T symbols (actual code)
    if symbol_parts[1] == 't' or symbol_parts[1] == 'T':
      module_list.append(symbol_parts[0])
  return (None, module_list)


def extract_individual_function(bitcode_module, extraction_path,
                                function_symbol):
  function_module_name = os.path.join(extraction_path, f'{function_symbol}.bc')
  extract_command_vector = [
      'llvm-extract', '-func', function_symbol, '-o', function_module_name
  ]
  try:
    with subprocess.Popen(
        extract_command_vector,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE) as extraction_process:
      stdout = extraction_process.communicate(
          input=bitcode_module)[0].decode('utf-8')
      if extraction_process.returncode != 0:
        logging.info(f'Failed to extract {function_symbol}')
        return (stdout.replace('\n', ''), None)
  except OSError:
    logging.info(f'Failed to extract {function_symbol} due to OSError')
    return ('oserror', None)

  return (None, function_module_name)


def get_run_passes_opt(bitcode_function_path):
  opt_command_vector = [
      'opt', bitcode_function_path, '-print-changed', '-passes=default<O3>',
      '-o', '/dev/null'
  ]
  try:
    opt_process = subprocess.run(
        opt_command_vector,
        encoding='UTF-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=OPT_TIMEOUT_SECONDS)
  except Exception:
    return ('timeout', None)
  if opt_process.returncode != 0:
    return (opt_process.stdout.replace('\n', ''), None)
  opt_process_lines = opt_process.stdout.split('\n')
  pass_indexes = {}
  passes = {}
  for opt_process_line in opt_process_lines:
    if opt_process_line[:3] == '***' and opt_process_line[-3:] == '***':
      # We're in a pass status line
      if opt_process_line[4:11] == 'IR Pass':
        # Anything starting with IR Pass gets ignored, so we can't do anything
        # with it.
        continue
      if opt_process_line[12:20] == 'At Start':
        # Ignore the starting IR
        continue
      pass_name = opt_process_line.split(' on ')[0][12:]
      pass_name = pass_name.split('After ')[1]
      ir_changed = opt_process_line[-13:-4] == 'no change'
      # Special case loop passes because they run once per loop rather than
      # once per function.
      if pass_name in pass_list_constants.LOOP_PASS_LIST:
        pass_name = pass_name + '1'
        if pass_name not in passes or not passes[pass_name]:
          passes[pass_name] = ir_changed
      elif pass_name in pass_indexes:
        pass_indexes[pass_name] += 1
        pass_name = f'{pass_name}{pass_indexes[pass_name]}'
      else:
        pass_indexes[pass_name] = 1
        pass_name = pass_name + '1'
      if ir_changed:
        passes[pass_name] = [False]
      else:
        passes[pass_name] = [True]
  return (None, passes)


def combine_statistics(function_a, function_b, fill_value=False):
  if function_a is None or function_a == {}:
    return function_b
  combined_statistics = function_a
  combined_statistics_length = len(combined_statistics[list(
      combined_statistics.keys())[0]])
  for function_statistic in list(
      set(list(function_a.keys()) + list(function_b.keys()))):
    if function_statistic in combined_statistics and function_statistic in function_b:
      combined_statistics[function_statistic].extend(
          function_b[function_statistic])
    elif function_statistic in function_b:
      combined_statistics[function_statistic] = [
          fill_value for i in range(0, combined_statistics_length)
      ]
      combined_statistics[function_statistic].extend(
          function_b[function_statistic])
    elif function_statistic in combined_statistics:
      function_b_statistics_length = len(function_b[list(function_b.keys())[0]])
      extra_values = [
          fill_value for i in range(0, function_b_statistics_length)
      ]
      combined_statistics[function_statistic].extend(extra_values)
  return combined_statistics


def get_function_properties(bitcode_function_path,
                            passes="forceattrs,print<func-properties>"):
  properties_dict = {}
  opt_command_vector = [
      'opt', f'-passes={passes}', bitcode_function_path,
      '-enable-detailed-function-properties', '-disable-output',
      '-force-remove-attribute=optnone'
  ]
  try:
    opt_process = subprocess.run(
        opt_command_vector,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        timeout=OPT_TIMEOUT_SECONDS)
  except subprocess.SubprocessError:
    return ('timeout', None)
  if opt_process.returncode != 0:
    return (opt_process.stdout.replace('\n', ''), None)
  output_lines = opt_process.stdout.split('\n')[1:-2]
  for output_line in output_lines:
    line_parts = output_line.split(': ')
    if len(line_parts) < 2:
      return ('invalid opt output', None)
    properties_dict[line_parts[0]] = [line_parts[1]]
  return (None, properties_dict)


def get_function_properties_module(bitcode_module, extra_passes=''):
  if extra_passes != '':
    extra_passes += ','
  properties_dict = {}
  opt_command_vector = [
      'opt', f'-passes={extra_passes}forceattrs,print<func-properties>',
      '-enable-detailed-function-properties', '-force-remove-attribute=optnone',
      '-disable-output', '-'
  ]
  with subprocess.Popen(
      opt_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      stdin=subprocess.PIPE) as opt_process:
    try:
      stdout = opt_process.communicate(
          input=bitcode_module, timeout=OPT_TIMEOUT_SECONDS)[0].decode('utf-8')
    except subprocess.TimeoutExpired:
      return ('timout', None)
    if opt_process.returncode != 0:
      return (stdout.replace('\n', ''), None)

    start_index = 0
    output_lines_raw = stdout.split('\n')[:-2]

    while start_index < len(output_lines_raw):
      if output_lines_raw[start_index].startswith('Printing'):
        break
      start_index += 1

    output_lines = output_lines_raw[start_index:]
    if len(output_lines) == 0:
      return ('no functions found in bitcode file', None)
    for output_line in output_lines:
      if output_line.startswith('Printing'):
        continue
      elif output_line == '':
        continue
      line_parts = output_line.split(': ')
      if line_parts[0] in properties_dict:
        properties_dict[line_parts[0]].append(line_parts[1])
      else:
        if len(line_parts) < 2:
          return ('invalid output from opt', None)
        properties_dict[line_parts[0]] = [line_parts[1]]
    return (None, properties_dict)


def get_instruction_counts(bitcode_module, additional_passes=''):
  properties_or_error = get_function_properties_module(bitcode_module,
                                                       additional_passes)
  if properties_or_error[0]:
    return None
  else:
    return [
        int(inst_count)
        for inst_count in properties_or_error[1]['TotalInstructionCount']
    ]


def get_instruction_histogram(bitcode_module, additional_passes=''):
  if additional_passes != '':
    additional_passes += ','
  instruction_histogram = {}
  opt_command_vector = [
      'opt', '-disable-output', f'-passes={additional_passes}instcount',
      '-stats'
  ]
  with subprocess.Popen(
      opt_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      stdin=subprocess.PIPE) as opt_process:
    try:
      output = opt_process.communicate(input=bitcode_module)[0].decode('utf-8')
    except subprocess.TimeoutExpired:
      return ('timeout', None)
    if opt_process.returncode != 0:
      return ('opt did not return with code zero', None)
    # Work on parsing the output
    output_lines = output.split('\n')
    # Skip the first five lines as they contain the stats header
    for output_line in output_lines[5:-2]:
      if 'instcount' not in output_line:
        continue
      output_line_parts = output_line.split()
      if len(output_line_parts) < 7:
        return ('opt returned invalid output', None)
      # Statistics line format is <count> <stat type> - number of <inst name>
      # This check skips all non instruction statistics also collected by the pass.
      if output_line_parts[6] != 'insts':
        continue
      instruction_name = output_line_parts[5]
      instruction_count = int(output_line_parts[0])
      instruction_histogram[instruction_name] = [instruction_count]
  return (None, instruction_histogram)


def get_instruction_histogram_from_file(bitcode_file_path):
  with open(bitcode_file_path, 'rb') as bitcode_file:
    return get_instruction_histogram(bitcode_file.read())


@ray.remote(num_cpus=1)
def get_function_statistics_batch(bitcode_module, function_symbols,
                                  statistics_type, module_path):
  statistics = []
  with tempfile.TemporaryDirectory() as extracted_functions_dir:
    for function_symbol in function_symbols:
      expected_extracted_function_path = extract_individual_function(
          bitcode_module, extracted_functions_dir, function_symbol)
      function_path = f'{module_path}:{function_symbol}'
      if expected_extracted_function_path[0]:
        statistics.append(
            (expected_extracted_function_path[0], None, function_path))
        continue
      bitcode_function_path = expected_extracted_function_path[1]
      if statistics_type == 'properties':
        function_statistics_expected = get_function_properties(
            bitcode_function_path)
      elif statistics_type == 'passes':
        function_statistics_expected = get_run_passes_opt(bitcode_function_path)
      elif statistics_type == 'post_opt_properties':
        function_statistics_expected = get_function_properties(
            bitcode_function_path,
            'forceattrs,default<O3>,print<func-properties>')
      elif statistics_type == 'instruction_distribution':
        function_statistics_expected = get_instruction_histogram_from_file(
            bitcode_function_path)
      if function_statistics_expected[0]:
        statistics.append(
            (function_statistics_expected[0], None, function_path))
      else:
        statistics.append(
            (None, function_statistics_expected[1], function_path))
  return statistics


def get_bitcode_module_function_statistics(bitcode_module, statistics_type,
                                           module_path):
  function_symbols_expected = get_function_symbols(bitcode_module)

  if function_symbols_expected[0]:
    return [(function_symbols_expected[0], None, module_path)]

  function_symbols = function_symbols_expected[1]

  statistics_futures = []
  batches = parallel.split_batches(function_symbols, BITCODE_FILE_CHUNK_SIZE)
  for batch in batches:
    statistics_futures.append(
        get_function_statistics_batch.remote(bitcode_module, batch,
                                             statistics_type, module_path))

  statistics_chunks = ray.get(statistics_futures)
  statistics = []
  for statistics_chunk in statistics_chunks:
    statistics.extend(statistics_chunk)
  return statistics


def test_parsing(bitcode_module):
  opt_command_vector = ['opt', '-', '-o', '/dev/null']
  with subprocess.Popen(
      opt_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      stdin=subprocess.PIPE) as opt_process:
    stdout = opt_process.communicate(
        input=bitcode_module, timeout=OPT_TIMEOUT_SECONDS)[0].decode('utf-8')
    return (stdout.replace('\n', ''), {
        'parseable': [opt_process.returncode == 0]
    })


def get_size(bitcode_module):
  return (None, {'size': [len(bitcode_module)]})


def get_textual_ir(bitcode_module):
  dis_command_vector = ['llvm-dis', '-']
  with subprocess.Popen(
      dis_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      stdin=subprocess.PIPE) as dis_process:
    try:
      output = dis_process.communicate(
          input=bitcode_module,
          timeout=LLVM_DIS_TIMEOUT_SECONDS)[0].decode('utf-8')
    except subprocess.TimeoutExpired:
      return ('timeout', None)
    if dis_process.returncode != 0:
      return ('llvm-dis returned code other than 0', None)
    return (None, output)


def get_size_text(bitcode_module):
  textual_ir_or_error = get_textual_ir(bitcode_module)
  if textual_ir_or_error[0]:
    return (textual_ir_or_error[0], None)
  return (None, {'size': [len(textual_ir_or_error[1])]})


def get_token_count(bitcode_module, vocab_path):
  textual_ir_or_error = get_textual_ir(bitcode_module)
  if textual_ir_or_error[0]:
    return (textual_ir_or_error[0], None)
  with tempfile.NamedTemporaryFile(
  ) as textual_ir_file, tempfile.NamedTemporaryFile() as tokenized_file:
    textual_ir_file.write(textual_ir_or_error[1].encode('utf-8'))
    fast_command_vector = [
        'fast', 'applybpe', tokenized_file.name, textual_ir_file.name,
        vocab_path
    ]
    try:
      fast_process = subprocess.run(
          fast_command_vector,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          timeout=FASTBPE_TIMEOUT_SECONDS)
      if fast_process.returncode != 0:
        return ('fastbpe returned non-zero exit code', None)
      output = tokenized_file.read().decode('utf-8')
    except subprocess.TimeoutExpired:
      return ('fastbpe timeout expired', None)
    return (None, output.count('@@'))


def get_hf_token_count(bitcode_module, tokenizer_json):
  textual_ir_or_error = get_textual_ir(bitcode_module)
  if textual_ir_or_error[0]:
    return (textual_ir_or_error[0], None)

  import sentencepiece as snp
  tokenizer_object = snp.SentencePieceProcessor(model_file=tokenizer_json)

  token_count = 0

  # Chunk the textual IR so that the tokenizer memory usage does not explode.
  # This is not really the optimal way to do things and does slightly impact
  # output accuracey (2-3% wrappinga at 10^6 from my testing). The number is
  # somewhat arbitrary, but seems to work for most corpora on a machine with
  # 96 threads/256GB of RAM.
  for string_part in textwrap.wrap(textual_ir_or_error[1], 5000000):
    token_count += len(tokenizer_object.encode(string_part))

  return (None, token_count)


def get_lowered_size(bitcode_module):
  # Run llc on the bitcode to lower to assembly
  llc_command_vector = ['llc', '-filetype=obj', '-']
  with subprocess.Popen(
      llc_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT) as llc_process:
    llc_output = llc_process.communicate(input=bitcode_module)[0]
    if llc_process.returncode != 0:
      return ('llc returned non-zero exit code', None)
  # Use llvm-size to measure the output size
  # Note that the format specified here actually impacts the output text size
  # as certain modes that LLVM aims to be compatible with count things differently.
  # --format=sysv seems to specifically count data contained in .txt sections, which
  # is what we're after.
  llvm_size_command_vector = ['llvm-size', '--format=sysv', '-']
  with subprocess.Popen(
      llvm_size_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT) as llvm_size_process:
    llvm_size_output = llvm_size_process.communicate(
        input=llc_output)[0].decode('utf-8')
  llvm_size_output_lines = llvm_size_output.split('\n')
  if len(llvm_size_output_lines) < 3:
    return ('llvm-size returned invalid output', None)
  if len(llvm_size_output_lines[2].split()) < 2:
    return ('llvm-size returned invalid output', None)
  return (None, int(llvm_size_output_lines[2].split()[1]))


def get_optimized_bitcode(bitcode_module):
  # Run the opt O3 pipeline on the module.
  opt_command_vector = ['opt', '-passes=default<O3>', '-']
  with subprocess.Popen(
      opt_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT) as opt_process:
    return opt_process.communicate(input=bitcode_module)[0]


def strip_debuginfo(bitcode_module):
  # Run opt -strip-debug to get rid of debug information.
  opt_command_vector = ['opt', '-strip-debug', '-']
  with subprocess.Popen(
      opt_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT) as opt_process:
    return opt_process.communicate(input=bitcode_module)[0]


def get_lowered_size_post_opt(bitcode_module):
  optimized_bc = get_optimized_bitcode(bitcode_module)
  return get_lowered_size(optimized_bc)


def get_call_names_pass_path():
  return shutil.which('libPrintCallNamesPass.so')


def get_call_names(bitcode_module):
  call_names_pass_path = get_call_names_pass_path()
  opt_command_vector = [
      'opt', '-load-pass-plugin', call_names_pass_path,
      '-passes=print<call-names>', '-disable-output', '-'
  ]
  with subprocess.Popen(
      opt_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT) as opt_process:
    opt_output = opt_process.communicate(
        input=bitcode_module)[0].decode('utf-8')
    if (opt_process.returncode != 0):
      return []
    return opt_output.split('\n')[:-1]


def get_defined_function_names(bitcode_module):
  opt_command_vector = [
      'opt', '-load-pass-plugin',
      get_call_names_pass_path(), '-passes=print<definition-names>',
      '-disable-output'
  ]
  with subprocess.Popen(
      opt_command_vector,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      stdin=subprocess.PIPE) as opt_process:
    try:
      stdout = opt_process.communicate(input=bitcode_module)[0].decode('utf-8')
    except subprocess.TimeoutExpired:
      return ('timeout', None)
    if opt_process.returncode != 0:
      return ('opt returned code other than 0', None)
    return (None, stdout.split('\n')[:-1])


def get_function_hashes(bitcode_module, additional_passes=''):
  if additional_passes != '':
    additional_passes = additional_passes + ','
  opt_hashing_vector = [
      'opt',
      f'-passes={additional_passes}forceattrs,print<structural-hash><detailed>',
      '-disable-output', '-', '-force-remove-attribute=optnone'
  ]
  with subprocess.Popen(
      opt_hashing_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT) as opt_process:
    try:
      opt_output = opt_process.communicate(
          input=bitcode_module, timeout=OPT_TIMEOUT_SECONDS)[0].decode('utf-8')
    except subprocess.TimeoutExpired:
      return ('timeout', None, None)
    except UnicodeDecodeError:
      return ('unicode error, opt returned invalid output', None, None)
    if opt_process.returncode != 0:
      return ('opt did not exit with code 0', None, None)
    function_hashes = {}
    output_lines = opt_output.split('\n')

    start_line_index = 0
    while start_line_index < len(output_lines):
      if output_lines[start_line_index].startswith("Module Hash:"):
        break
      start_line_index += 1

    if start_line_index == output_lines:
      return ('invalid output from opt - did not find module hash line.', None)

    module_hash_line_parts = output_lines[start_line_index].split()
    module_hash = module_hash_line_parts[2]
    for output_line in output_lines[(start_line_index + 1):-1]:
      output_line_parts = output_line.split()
      if len(output_line_parts) < 4:
        return ('invalid output from opt', None)
      function_name = output_line_parts[-3]
      function_hash = output_line_parts[-1]
      function_hashes[function_name] = function_hash
    return (None, function_hashes, module_hash)


@ray.remote(num_cpus=1)
def get_module_statistics_batch(project_dir,
                                module_paths,
                                statistics_type,
                                filter='none',
                                extra_properties={}):
  statistics = []
  for relative_module_path in module_paths:
    bitcode_file = dataset_corpus.load_file_from_corpus(project_dir,
                                                        relative_module_path)
    if filter != 'none':
      command_line_path = os.path.splitext(relative_module_path)[0] + '.cmd'
      command_line_raw = dataset_corpus.load_file_from_corpus(
          project_dir, command_line_path)
      if command_line_raw is None:
        continue

      command_line = command_line_raw.decode('utf-8')
      # This is a very hacky heuristic, mostly based on how many include paths
      # the driver tries to add to the frontend command line. Might need to be
      # fixed in the future for portability.
      if filter == 'cpp' and command_line.count('c++') <= 1:
        continue
      elif filter == 'c' and command_line.count('c++') > 1:
        continue

    module_path = f'{project_dir}:{relative_module_path}'
    if statistics_type == 'parsing':
      parse_result = test_parsing(bitcode_file)
      if parse_result[1]:
        statistics.append((None, parse_result[1], module_path))
      else:
        statistics.append((parse_result[0], parse_result[1], module_path))
    elif statistics_type == 'module_size':
      if bitcode_file is None:
        continue
      statistics.append((None, get_size(bitcode_file)[1], module_path))
    elif statistics_type == 'module_size_text':
      text_size_or_error = get_size_text(bitcode_file)
      if text_size_or_error[0]:
        statistics.append((text_size_or_error[0], None, module_path))
      else:
        statistics.append((None, text_size_or_error[1], module_path))
    elif statistics_type == 'get_lowered_size':
      lowered_size_or_error = get_lowered_size(bitcode_file)
      if lowered_size_or_error[0] is not None:
        statistics.append((lowered_size_or_error[0], None, module_path))
        continue
      lowered_size = lowered_size_or_error[1]
      wrapped_result = {'lowered_size': [lowered_size]}
      statistics.append((None, wrapped_result, module_path))
    elif statistics_type == 'get_opt_lowered_size':
      post_opt_lowered_size_or_error = get_lowered_size_post_opt(bitcode_file)
      if post_opt_lowered_size_or_error[0] is not None:
        statistics.append(
            (post_opt_lowered_size_or_error[0], None, module_path))
        continue
      post_opt_lowered_size = post_opt_lowered_size_or_error[1]
      wrapped_result = {'post_opt_lowered_size': [post_opt_lowered_size]}
      statistics.append((None, wrapped_result, module_path))
    elif statistics_type == 'call_names':
      for call_name in get_call_names(bitcode_file):
        call_names_wrapped = {'call_names': [call_name]}
        statistics.append((None, call_names_wrapped, module_path))
    elif statistics_type == 'function_hashes' or statistics_type == 'post_O3_function_hashes':
      additional_passes = '' if statistics_type == 'function_hashes' else 'default<O3>'
      function_hashes_or_error = get_function_hashes(bitcode_file,
                                                     additional_passes)
      if function_hashes_or_error[0]:
        statistics.append((function_hashes_or_error[0], None, module_path))
        continue
      function_hashes = function_hashes_or_error[1]
      for function_name in function_hashes:
        hash_wrapped = {'function_hashes': [function_hashes[function_name]]}
        statistics.append(
            (None, hash_wrapped, f'{module_path}:{function_name}'))
    elif statistics_type == 'module_hashes':
      module_hash_or_error = get_function_hashes(bitcode_file)
      if module_hash_or_error[0]:
        statistics.append((module_hash_or_error[0], None, module_path))
      else:
        hash_wrapped = {'module_hashes': [module_hash_or_error[2]]}
        statistics.append((None, hash_wrapped, module_path))
    elif statistics_type == 'module_properties' or statistics_type == 'module_properties_O3':
      additional_passes = '' if statistics_type == 'module_properties' else 'default<O3>'
      properties_tuple = get_function_properties_module(bitcode_file,
                                                        additional_passes)
      if properties_tuple[0]:
        statistics.append((properties_tuple[0], None, module_path))
      else:
        statistics.append((None, properties_tuple[1], module_path))
    elif statistics_type == 'module_instruction_distribution' or \
      statistics_type == 'module_instruction_distribution_O3':
      additional_passes = '' if statistics_type == 'module_instruction_distribution' else 'default<O3>'
      instruction_hist_or_error = get_instruction_histogram(
          bitcode_file, additional_passes)
      if instruction_hist_or_error[0]:
        statistics.append((instruction_hist_or_error[0], None, module_path))
      else:
        statistics.append((None, instruction_hist_or_error[1], module_path))
    elif statistics_type == 'defined_function_names':
      function_names_or_error = get_defined_function_names(bitcode_file)
      if function_names_or_error[0]:
        statistics.append((function_names_or_error[0], None, module_path))
      else:
        for defined_function_name in function_names_or_error[1]:
          function_name_wrapped = {'defined_function': [defined_function_name]}
          statistics.append((None, function_name_wrapped, module_path))
    elif statistics_type == 'token_count':
      token_count_or_error = get_token_count(bitcode_file,
                                             extra_properties['bpe_vocab_path'])
      if token_count_or_error[0]:
        statistics.append((token_count_or_error[0], None, module_path))
      else:
        token_count_wrapped = {'token_count': [token_count_or_error[1]]}
        statistics.append((None, token_count_wrapped, module_path))
    elif statistics_type == 'hf_token_count':
      token_count_or_error = get_hf_token_count(
          bitcode_file, extra_properties['bpe_vocab_path'])
      token_count_wrapped = {'hf_token_count': [token_count_or_error[1]]}
      statistics.append((None, token_count_wrapped, module_path))
  return statistics


def get_tokenization(bitcode_module):
  tokenizer_command_vector = ['llvm-tokenizer', '-output-mode=json', '-']
  with subprocess.Popen(
      tokenizer_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE) as tokenizer_process:
    try:
      stdout = tokenizer_process.communicate(input=bitcode_module)[0]
      return json.loads(stdout)
    except json.JSONDecodeError:
      # TODO(boomanaiden154): This is failing pretty often. Get more debug
      # information (like file path) into these logs so we can do downstream
      # analysis.
      logging.warning('Failed to decode JSON')
      return {}


def get_serialized_tokenization(bitcode_module, int_constants_path):
  tokenizer_command_vector = [
      'llvm-tokenizer', '-output-mode=json', '-mode=serialize',
      f'-int-constants-list={int_constants_path}'
  ]
  with subprocess.Popen(
      tokenizer_command_vector,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE) as tokenizer_process:
    try:
      stdout = tokenizer_process.communicate(input=bitcode_module)[0]
      tokenizer_output = json.loads(stdout)

      tokenization = []

      for function in tokenizer_output['functions']:
        tokenization += function['tokens']

      return tokenization
    except json.JSONDecodeError:
      logging.warning('Failed to decode JSON')
      return []
