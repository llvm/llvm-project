from __future__ import print_function

import argparse
import copy
import glob
import itertools
import os
import re
import subprocess
import sys
import shlex

from typing import List

##### Common utilities for update_*test_checks.py


_verbose = False
_prefix_filecheck_ir_name = ''

class Regex(object):
  """Wrap a compiled regular expression object to allow deep copy of a regexp.
  This is required for the deep copy done in do_scrub.

  """
  def __init__(self, regex):
    self.regex = regex

  def __deepcopy__(self, memo):
    result = copy.copy(self)
    result.regex = self.regex
    return result

  def search(self, line):
    return self.regex.search(line)

  def sub(self, repl, line):
    return self.regex.sub(repl, line)

  def pattern(self):
    return self.regex.pattern

  def flags(self):
    return self.regex.flags

class Filter(Regex):
  """Augment a Regex object with a flag indicating whether a match should be
    added (!is_filter_out) or removed (is_filter_out) from the generated checks.

  """
  def __init__(self, regex, is_filter_out):
    super(Filter, self).__init__(regex)
    self.is_filter_out = is_filter_out

  def __deepcopy__(self, memo):
    result = copy.deepcopy(super(Filter, self), memo)
    result.is_filter_out = copy.deepcopy(self.is_filter_out, memo)
    return result

def parse_commandline_args(parser):
  class RegexAction(argparse.Action):
    """Add a regular expression option value to a list of regular expressions.
    This compiles the expression, wraps it in a Regex and adds it to the option
    value list."""
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
      if nargs is not None:
        raise ValueError('nargs not allowed')
      super(RegexAction, self).__init__(option_strings, dest, **kwargs)

    def do_call(self, namespace, values, flags):
      value_list = getattr(namespace, self.dest)
      if value_list is None:
        value_list = []

      try:
        value_list.append(Regex(re.compile(values, flags)))
      except re.error as error:
        raise ValueError('{}: Invalid regular expression \'{}\' ({})'.format(
          option_string, error.pattern, error.msg))

      setattr(namespace, self.dest, value_list)

    def __call__(self, parser, namespace, values, option_string=None):
      self.do_call(namespace, values, 0)

  class FilterAction(RegexAction):
    """Add a filter to a list of filter option values."""
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
      super(FilterAction, self).__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
      super(FilterAction, self).__call__(parser, namespace, values, option_string)

      value_list = getattr(namespace, self.dest)

      is_filter_out = ( option_string == '--filter-out' )

      value_list[-1] = Filter(value_list[-1].regex, is_filter_out)

      setattr(namespace, self.dest, value_list)

  filter_group = parser.add_argument_group(
    'filtering',
    """Filters are applied to each output line according to the order given. The
    first matching filter terminates filter processing for that current line.""")

  filter_group.add_argument('--filter', action=FilterAction, dest='filters',
                            metavar='REGEX',
                            help='Only include lines matching REGEX (may be specified multiple times)')
  filter_group.add_argument('--filter-out', action=FilterAction, dest='filters',
                            metavar='REGEX',
                            help='Exclude lines matching REGEX')

  parser.add_argument('--include-generated-funcs', action='store_true',
                      help='Output checks for functions not in source')
  parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show verbose output')
  parser.add_argument('-u', '--update-only', action='store_true',
                      help='Only update test if it was already autogened')
  parser.add_argument('--force-update', action='store_true',
                      help='Update test even if it was autogened by a different script')
  parser.add_argument('--enable', action='store_true', dest='enabled', default=True,
                       help='Activate CHECK line generation from this point forward')
  parser.add_argument('--disable', action='store_false', dest='enabled',
                      help='Deactivate CHECK line generation from this point forward')
  parser.add_argument('--replace-value-regex', nargs='+', default=[],
                      help='List of regular expressions to replace matching value names')
  parser.add_argument('--prefix-filecheck-ir-name', default='',
                      help='Add a prefix to FileCheck IR value names to avoid conflicts with scripted names')
  parser.add_argument('--global-value-regex', nargs='+', default=[],
                      help='List of regular expressions that a global value declaration must match to generate a check (has no effect if checking globals is not enabled)')
  parser.add_argument('--global-hex-value-regex', nargs='+', default=[],
                      help='List of regular expressions such that, for matching global value declarations, literal integer values should be encoded in hex in the associated FileCheck directives')
  # FIXME: in 3.9, we can use argparse.BooleanOptionalAction. At that point,
  # we need to rename the flag to just -generate-body-for-unused-prefixes.
  parser.add_argument('--no-generate-body-for-unused-prefixes',
                      action='store_false',
                      dest='gen_unused_prefix_body',
                      default=True,
                      help='Generate a function body that always matches for unused prefixes. This is useful when unused prefixes are desired, and it avoids needing to annotate each FileCheck as allowing them.')
  args = parser.parse_args()
  global _verbose, _global_value_regex, _global_hex_value_regex
  _verbose = args.verbose
  _global_value_regex = args.global_value_regex
  _global_hex_value_regex = args.global_hex_value_regex
  return args


class InputLineInfo(object):
  def __init__(self, line, line_number, args, argv):
    self.line = line
    self.line_number = line_number
    self.args = args
    self.argv = argv


class TestInfo(object):
  def __init__(self, test, parser, script_name, input_lines, args, argv,
               comment_prefix, argparse_callback):
    self.parser = parser
    self.argparse_callback = argparse_callback
    self.path = test
    self.args = args
    if args.prefix_filecheck_ir_name:
      global _prefix_filecheck_ir_name
      _prefix_filecheck_ir_name = args.prefix_filecheck_ir_name
    self.argv = argv
    self.input_lines = input_lines
    self.run_lines = find_run_lines(test, self.input_lines)
    self.comment_prefix = comment_prefix
    if self.comment_prefix is None:
      if self.path.endswith('.mir'):
        self.comment_prefix = '#'
      else:
        self.comment_prefix = ';'
    self.autogenerated_note_prefix = self.comment_prefix + ' ' + UTC_ADVERT
    self.test_autogenerated_note = self.autogenerated_note_prefix + script_name
    self.test_autogenerated_note += get_autogennote_suffix(parser, self.args)
    self.test_unused_note = self.comment_prefix + self.comment_prefix + ' ' + UNUSED_NOTE

  def ro_iterlines(self):
    for line_num, input_line in enumerate(self.input_lines):
      args, argv = check_for_command(input_line, self.parser,
                                     self.args, self.argv, self.argparse_callback)
      yield InputLineInfo(input_line, line_num, args, argv)

  def iterlines(self, output_lines):
    output_lines.append(self.test_autogenerated_note)
    for line_info in self.ro_iterlines():
      input_line = line_info.line
      # Discard any previous script advertising.
      if input_line.startswith(self.autogenerated_note_prefix):
        continue
      self.args = line_info.args
      self.argv = line_info.argv
      if not self.args.enabled:
        output_lines.append(input_line)
        continue
      yield line_info

  def get_checks_for_unused_prefixes(self, run_list, used_prefixes: List[str]) -> List[str]:
    run_list = [element for element in run_list if element[0] is not None]
    unused_prefixes = set([
        prefix for sublist in run_list for prefix in sublist[0]
    ]).difference(set(used_prefixes))

    ret = []
    if not unused_prefixes:
      return ret
    ret.append(self.test_unused_note)
    for unused in sorted(unused_prefixes):
      ret.append('{comment} {prefix}: {match_everything}'.format(
        comment=self.comment_prefix,
        prefix=unused,
        match_everything=r"""{{.*}}"""
      ))
    return ret

def itertests(test_patterns, parser, script_name, comment_prefix=None, argparse_callback=None):
  for pattern in test_patterns:
    # On Windows we must expand the patterns ourselves.
    tests_list = glob.glob(pattern)
    if not tests_list:
      warn("Test file pattern '%s' was not found. Ignoring it." % (pattern,))
      continue
    for test in tests_list:
      with open(test) as f:
        input_lines = [l.rstrip() for l in f]
      args = parser.parse_args()
      if argparse_callback is not None:
        argparse_callback(args)
      argv = sys.argv[:]
      first_line = input_lines[0] if input_lines else ""
      if UTC_ADVERT in first_line:
        if script_name not in first_line and not args.force_update:
          warn("Skipping test which wasn't autogenerated by " + script_name, test)
          continue
        args, argv = check_for_command(first_line, parser, args, argv, argparse_callback)
      elif args.update_only:
        assert UTC_ADVERT not in first_line
        warn("Skipping test which isn't autogenerated: " + test)
        continue
      final_input_lines = []
      for l in input_lines:
        if UNUSED_NOTE in l:
          break
        final_input_lines.append(l)
      yield TestInfo(test, parser, script_name, final_input_lines, args, argv,
                     comment_prefix, argparse_callback)


def should_add_line_to_output(input_line, prefix_set, skip_global_checks = False, comment_marker = ';'):
  # Skip any blank comment lines in the IR.
  if not skip_global_checks and input_line.strip() == comment_marker:
    return False
  # Skip a special double comment line we use as a separator.
  if input_line.strip() == comment_marker + SEPARATOR:
    return False
  # Skip any blank lines in the IR.
  #if input_line.strip() == '':
  #  return False
  # And skip any CHECK lines. We're building our own.
  m = CHECK_RE.match(input_line)
  if m and m.group(1) in prefix_set:
    if skip_global_checks:
      global_ir_value_re = re.compile(r'\[\[', flags=(re.M))
      return not global_ir_value_re.search(input_line)
    return False

  return True

# Perform lit-like substitutions
def getSubstitutions(sourcepath):
  sourcedir = os.path.dirname(sourcepath)
  return [('%s', sourcepath),
          ('%S', sourcedir),
          ('%p', sourcedir),
          ('%{pathsep}', os.pathsep)]

def applySubstitutions(s, substitutions):
  for a,b in substitutions:
    s = s.replace(a, b)
  return s

# Invoke the tool that is being tested.
def invoke_tool(exe, cmd_args, ir, preprocess_cmd=None, verbose=False):
  with open(ir) as ir_file:
    substitutions = getSubstitutions(ir)

    # TODO Remove the str form which is used by update_test_checks.py and
    # update_llc_test_checks.py
    # The safer list form is used by update_cc_test_checks.py
    if preprocess_cmd:
      # Allow pre-processing the IR file (e.g. using sed):
      assert isinstance(preprocess_cmd, str)  # TODO: use a list instead of using shell
      preprocess_cmd = applySubstitutions(preprocess_cmd, substitutions).strip()
      if verbose:
        print('Pre-processing input file: ', ir, " with command '",
              preprocess_cmd, "'", sep="", file=sys.stderr)
      # Python 2.7 doesn't have subprocess.DEVNULL:
      with open(os.devnull, 'w') as devnull:
        pp = subprocess.Popen(preprocess_cmd, shell=True, stdin=devnull,
                              stdout=subprocess.PIPE)
        ir_file = pp.stdout

    if isinstance(cmd_args, list):
      args = [applySubstitutions(a, substitutions) for a in cmd_args]
      stdout = subprocess.check_output([exe] + args, stdin=ir_file)
    else:
      stdout = subprocess.check_output(exe + ' ' + applySubstitutions(cmd_args, substitutions),
                                       shell=True, stdin=ir_file)
    if sys.version_info[0] > 2:
      # FYI, if you crashed here with a decode error, your run line probably
      # results in bitcode or other binary format being written to the pipe.
      # For an opt test, you probably want to add -S or -disable-output.
      stdout = stdout.decode()
  # Fix line endings to unix CR style.
  return stdout.replace('\r\n', '\n')

##### LLVM IR parser
RUN_LINE_RE = re.compile(r'^\s*(?://|[;#])\s*RUN:\s*(.*)$')
CHECK_PREFIX_RE = re.compile(r'--?check-prefix(?:es)?[= ](\S+)')
PREFIX_RE = re.compile('^[a-zA-Z0-9_-]+$')
CHECK_RE = re.compile(r'^\s*(?://|[;#])\s*([^:]+?)(?:-NEXT|-NOT|-DAG|-LABEL|-SAME|-EMPTY)?:')

UTC_ARGS_KEY = 'UTC_ARGS:'
UTC_ARGS_CMD = re.compile(r'.*' + UTC_ARGS_KEY + '\s*(?P<cmd>.*)\s*$')
UTC_ADVERT = 'NOTE: Assertions have been autogenerated by '
UNUSED_NOTE = 'NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:'

OPT_FUNCTION_RE = re.compile(
    r'^(\s*;\s*Function\sAttrs:\s(?P<attrs>[\w\s():,]+?))?\s*define\s+(?:internal\s+)?[^@]*@(?P<func>[\w.$-]+?)\s*'
    r'(?P<args_and_sig>\((\)|(.*?[\w.-]+?)\))[^{]*\{)\n(?P<body>.*?)^\}$',
    flags=(re.M | re.S))

ANALYZE_FUNCTION_RE = re.compile(
    r'^\s*\'(?P<analysis>[\w\s-]+?)\'\s+for\s+function\s+\'(?P<func>[\w.$-]+?)\':'
    r'\s*\n(?P<body>.*)$',
    flags=(re.X | re.S))

LV_DEBUG_RE = re.compile(
    r'^\s*\'(?P<func>[\w.$-]+?)\'[^\n]*'
    r'\s*\n(?P<body>.*)$',
    flags=(re.X | re.S))

IR_FUNCTION_RE = re.compile(r'^\s*define\s+(?:internal\s+)?[^@]*@"?([\w.$-]+)"?\s*\(')
TRIPLE_IR_RE = re.compile(r'^\s*target\s+triple\s*=\s*"([^"]+)"$')
TRIPLE_ARG_RE = re.compile(r'-mtriple[= ]([^ ]+)')
MARCH_ARG_RE = re.compile(r'-march[= ]([^ ]+)')
DEBUG_ONLY_ARG_RE = re.compile(r'-debug-only[= ]([^ ]+)')

SCRUB_LEADING_WHITESPACE_RE = re.compile(r'^(\s+)')
SCRUB_WHITESPACE_RE = re.compile(r'(?!^(|  \w))[ \t]+', flags=re.M)
SCRUB_TRAILING_WHITESPACE_RE = re.compile(r'[ \t]+$', flags=re.M)
SCRUB_TRAILING_WHITESPACE_TEST_RE = SCRUB_TRAILING_WHITESPACE_RE
SCRUB_TRAILING_WHITESPACE_AND_ATTRIBUTES_RE = re.compile(r'([ \t]|(#[0-9]+))+$', flags=re.M)
SCRUB_KILL_COMMENT_RE = re.compile(r'^ *#+ +kill:.*\n')
SCRUB_LOOP_COMMENT_RE = re.compile(
    r'# =>This Inner Loop Header:.*|# in Loop:.*', flags=re.M)
SCRUB_TAILING_COMMENT_TOKEN_RE = re.compile(r'(?<=\S)+[ \t]*#$', flags=re.M)

SEPARATOR = '.'

def error(msg, test_file=None):
  if test_file:
    msg = '{}: {}'.format(msg, test_file)
  print('ERROR: {}'.format(msg), file=sys.stderr)

def warn(msg, test_file=None):
  if test_file:
    msg = '{}: {}'.format(msg, test_file)
  print('WARNING: {}'.format(msg), file=sys.stderr)

def debug(*args, **kwargs):
  # Python2 does not allow def debug(*args, file=sys.stderr, **kwargs):
  if 'file' not in kwargs:
    kwargs['file'] = sys.stderr
  if _verbose:
    print(*args, **kwargs)

def find_run_lines(test, lines):
  debug('Scanning for RUN lines in test file:', test)
  raw_lines = [m.group(1)
               for m in [RUN_LINE_RE.match(l) for l in lines] if m]
  run_lines = [raw_lines[0]] if len(raw_lines) > 0 else []
  for l in raw_lines[1:]:
    if run_lines[-1].endswith('\\'):
      run_lines[-1] = run_lines[-1].rstrip('\\') + ' ' + l
    else:
      run_lines.append(l)
  debug('Found {} RUN lines in {}:'.format(len(run_lines), test))
  for l in run_lines:
    debug('  RUN: {}'.format(l))
  return run_lines

def get_triple_from_march(march):
  triples = {
      'amdgcn': 'amdgcn',
      'r600': 'r600',
      'mips': 'mips',
      'sparc': 'sparc',
      'hexagon': 'hexagon',
      've': 've',
  }
  for prefix, triple in triples.items():
    if march.startswith(prefix):
      return triple
  print("Cannot find a triple. Assume 'x86'", file=sys.stderr)
  return 'x86'

def apply_filters(line, filters):
  has_filter = False
  for f in filters:
    if not f.is_filter_out:
      has_filter = True
    if f.search(line):
      return False if f.is_filter_out else True
  # If we only used filter-out, keep the line, otherwise discard it since no
  # filter matched.
  return False if has_filter else True

def do_filter(body, filters):
  return body if not filters else '\n'.join(filter(
    lambda line: apply_filters(line, filters), body.splitlines()))

def scrub_body(body):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  body = SCRUB_WHITESPACE_RE.sub(r' ', body)
  # Expand the tabs used for indentation.
  body = str.expandtabs(body, 2)
  # Strip trailing whitespace.
  body = SCRUB_TRAILING_WHITESPACE_TEST_RE.sub(r'', body)
  return body

def do_scrub(body, scrubber, scrubber_args, extra):
  if scrubber_args:
    local_args = copy.deepcopy(scrubber_args)
    local_args[0].extra_scrub = extra
    return scrubber(body, *local_args)
  return scrubber(body, *scrubber_args)

# Build up a dictionary of all the function bodies.
class function_body(object):
  def __init__(self, string, extra, args_and_sig, attrs, func_name_separator):
    self.scrub = string
    self.extrascrub = extra
    self.args_and_sig = args_and_sig
    self.attrs = attrs
    self.func_name_separator = func_name_separator
  def is_same_except_arg_names(self, extrascrub, args_and_sig, attrs, is_backend):
    arg_names = set()
    def drop_arg_names(match):
      arg_names.add(match.group(variable_group_in_ir_value_match))
      if match.group(attribute_group_in_ir_value_match):
        attr = match.group(attribute_group_in_ir_value_match)
      else:
        attr = ''
      return match.group(1) + attr + match.group(match.lastindex)
    def repl_arg_names(match):
      if match.group(variable_group_in_ir_value_match) is not None and match.group(variable_group_in_ir_value_match) in arg_names:
        return match.group(1) + match.group(match.lastindex)
      return match.group(1) + match.group(2) + match.group(match.lastindex)
    if self.attrs != attrs:
      return False
    ans0 = IR_VALUE_RE.sub(drop_arg_names, self.args_and_sig)
    ans1 = IR_VALUE_RE.sub(drop_arg_names, args_and_sig)
    if ans0 != ans1:
      return False
    if is_backend:
      # Check without replacements, the replacements are not applied to the
      # body for backend checks.
      return self.extrascrub == extrascrub

    es0 = IR_VALUE_RE.sub(repl_arg_names, self.extrascrub)
    es1 = IR_VALUE_RE.sub(repl_arg_names, extrascrub)
    es0 = SCRUB_IR_COMMENT_RE.sub(r'', es0)
    es1 = SCRUB_IR_COMMENT_RE.sub(r'', es1)
    return es0 == es1

  def __str__(self):
    return self.scrub

class FunctionTestBuilder:
  def __init__(self, run_list, flags, scrubber_args, path):
    self._verbose = flags.verbose
    self._record_args = flags.function_signature
    self._check_attributes = flags.check_attributes
    # Strip double-quotes if input was read by UTC_ARGS
    self._filters = list(map(lambda f: Filter(re.compile(f.pattern().strip('"'),
                                                         f.flags()),
                                              f.is_filter_out),
                             flags.filters)) if flags.filters else []
    self._scrubber_args = scrubber_args
    self._path = path
    # Strip double-quotes if input was read by UTC_ARGS
    self._replace_value_regex = list(map(lambda x: x.strip('"'), flags.replace_value_regex))
    self._func_dict = {}
    self._func_order = {}
    self._global_var_dict = {}
    self._processed_prefixes = set()
    for tuple in run_list:
      for prefix in tuple[0]:
        self._func_dict.update({prefix: dict()})
        self._func_order.update({prefix: []})
        self._global_var_dict.update({prefix: dict()})

  def finish_and_get_func_dict(self):
    for prefix in self.get_failed_prefixes():
      warn('Prefix %s had conflicting output from different RUN lines for all functions in test %s' % (prefix,self._path,))
    return self._func_dict

  def func_order(self):
    return self._func_order

  def global_var_dict(self):
    return self._global_var_dict

  def is_filtered(self):
    return bool(self._filters)

  def process_run_line(self, function_re, scrubber, raw_tool_output, prefixes, is_backend):
    build_global_values_dictionary(self._global_var_dict, raw_tool_output, prefixes)
    for m in function_re.finditer(raw_tool_output):
      if not m:
        continue
      func = m.group('func')
      body = m.group('body')
      # func_name_separator is the string that is placed right after function name at the
      # beginning of assembly function definition. In most assemblies, that is just a
      # colon: `foo:`. But, for example, in nvptx it is a brace: `foo(`. If is_backend is
      # False, just assume that separator is an empty string.
      if is_backend:
        # Use ':' as default separator.
        func_name_separator = m.group('func_name_separator') if 'func_name_separator' in m.groupdict() else ':'
      else:
        func_name_separator = ''
      attrs = m.group('attrs') if self._check_attributes else ''
      # Determine if we print arguments, the opening brace, or nothing after the
      # function name
      if self._record_args and 'args_and_sig' in m.groupdict():
        args_and_sig = scrub_body(m.group('args_and_sig').strip())
      elif 'args_and_sig' in m.groupdict():
        args_and_sig = '('
      else:
        args_and_sig = ''
      filtered_body = do_filter(body, self._filters)
      scrubbed_body = do_scrub(filtered_body, scrubber, self._scrubber_args,
                               extra=False)
      scrubbed_extra = do_scrub(filtered_body, scrubber, self._scrubber_args,
                                extra=True)
      if 'analysis' in m.groupdict():
        analysis = m.group('analysis')
        if analysis.lower() != 'cost model analysis':
          warn('Unsupported analysis mode: %r!' % (analysis,))
      if func.startswith('stress'):
        # We only use the last line of the function body for stress tests.
        scrubbed_body = '\n'.join(scrubbed_body.splitlines()[-1:])
      if self._verbose:
        print('Processing function: ' + func, file=sys.stderr)
        for l in scrubbed_body.splitlines():
          print('  ' + l, file=sys.stderr)
      for prefix in prefixes:
        # Replace function names matching the regex.
        for regex in self._replace_value_regex:
          # Pattern that matches capture groups in the regex in leftmost order.
          group_regex = re.compile(r'\(.*?\)')
          # Replace function name with regex.
          match = re.match(regex, func)
          if match:
            func_repl = regex
            # Replace any capture groups with their matched strings.
            for g in match.groups():
              func_repl = group_regex.sub(re.escape(g), func_repl, count=1)
            func = re.sub(func_repl, '{{' + func_repl + '}}', func)

          # Replace all calls to regex matching functions.
          matches = re.finditer(regex, scrubbed_body)
          for match in matches:
            func_repl = regex
            # Replace any capture groups with their matched strings.
            for g in match.groups():
              func_repl = group_regex.sub(re.escape(g), func_repl, count=1)
            # Substitute function call names that match the regex with the same
            # capture groups set.
            scrubbed_body = re.sub(func_repl, '{{' + func_repl + '}}',
                                   scrubbed_body)

        if func in self._func_dict[prefix]:
          if (self._func_dict[prefix][func] is not None and
              (str(self._func_dict[prefix][func]) != scrubbed_body or
               self._func_dict[prefix][func].args_and_sig != args_and_sig or
               self._func_dict[prefix][func].attrs != attrs)):
            if self._func_dict[prefix][func].is_same_except_arg_names(
                scrubbed_extra,
                args_and_sig,
                attrs,
                is_backend):
              self._func_dict[prefix][func].scrub = scrubbed_extra
              self._func_dict[prefix][func].args_and_sig = args_and_sig
            else:
              # This means a previous RUN line produced a body for this function
              # that is different from the one produced by this current RUN line,
              # so the body can't be common across RUN lines. We use None to
              # indicate that.
              self._func_dict[prefix][func] = None
        else:
          if prefix not in self._processed_prefixes:
            self._func_dict[prefix][func] = function_body(
                scrubbed_body, scrubbed_extra, args_and_sig, attrs,
                func_name_separator)
            self._func_order[prefix].append(func)
          else:
            # An earlier RUN line used this check prefixes but didn't produce
            # a body for this function. This happens in Clang tests that use
            # preprocesser directives to exclude individual functions from some
            # RUN lines.
            self._func_dict[prefix][func] = None

  def processed_prefixes(self, prefixes):
    """
    Mark a set of prefixes as having had at least one applicable RUN line fully
    processed. This is used to filter out function bodies that don't have
    outputs for all RUN lines.
    """
    self._processed_prefixes.update(prefixes)

  def get_failed_prefixes(self):
    # This returns the list of those prefixes that failed to match any function,
    # because there were conflicting bodies produced by different RUN lines, in
    # all instances of the prefix.
    for prefix in self._func_dict:
      if (self._func_dict[prefix] and
          (not [fct for fct in self._func_dict[prefix]
                if self._func_dict[prefix][fct] is not None])):
        yield prefix


##### Generator of LLVM IR CHECK lines

SCRUB_IR_COMMENT_RE = re.compile(r'\s*;.*')

# TODO: We should also derive check lines for global, debug, loop declarations, etc..

class NamelessValue:
  def __init__(self, check_prefix, check_key, ir_prefix, global_ir_prefix, global_ir_prefix_regexp,
               ir_regexp, global_ir_rhs_regexp, is_before_functions, *,
               is_number=False, replace_number_with_counter=False):
    self.check_prefix = check_prefix
    self.check_key = check_key
    self.ir_prefix = ir_prefix
    self.global_ir_prefix = global_ir_prefix
    self.global_ir_prefix_regexp = global_ir_prefix_regexp
    self.ir_regexp = ir_regexp
    self.global_ir_rhs_regexp = global_ir_rhs_regexp
    self.is_before_functions = is_before_functions
    self.is_number = is_number
    # Some variable numbers (e.g. MCINST1234) will change based on unrelated
    # modifications to LLVM, replace those with an incrementing counter.
    self.replace_number_with_counter = replace_number_with_counter
    self.variable_mapping = {}

  # Return true if this kind of IR value is "local", basically if it matches '%{{.*}}'.
  def is_local_def_ir_value_match(self, match):
    return self.ir_prefix == '%'

  # Return true if this kind of IR value is "global", basically if it matches '#{{.*}}'.
  def is_global_scope_ir_value_match(self, match):
    return self.global_ir_prefix is not None

  # Return the IR prefix and check prefix we use for this kind or IR value,
  # e.g., (%, TMP) for locals.
  def get_ir_prefix_from_ir_value_match(self, match):
    if self.ir_prefix and match.group(0).strip().startswith(self.ir_prefix):
      return self.ir_prefix, self.check_prefix
    return self.global_ir_prefix, self.check_prefix

  # Return the IR regexp we use for this kind or IR value, e.g., [\w.-]+? for locals
  def get_ir_regex_from_ir_value_re_match(self, match):
    # for backwards compatibility we check locals with '.*'
    if self.is_local_def_ir_value_match(match):
      return '.*'
    if self.ir_prefix and match.group(0).strip().startswith(self.ir_prefix):
      return self.ir_regexp
    return self.global_ir_prefix_regexp

  # Create a FileCheck variable name based on an IR name.
  def get_value_name(self, var: str, check_prefix: str):
    var = var.replace('!', '')
    if self.replace_number_with_counter:
      assert var.isdigit(), var
      replacement = self.variable_mapping.get(var, None)
      if replacement is None:
        # Replace variable with an incrementing counter
        replacement = str(len(self.variable_mapping) + 1)
        self.variable_mapping[var] = replacement
      var = replacement
    # This is a nameless value, prepend check_prefix.
    if var.isdigit():
      var = check_prefix + var
    else:
      # This is a named value that clashes with the check_prefix, prepend with
      # _prefix_filecheck_ir_name, if it has been defined.
      if may_clash_with_default_check_prefix_name(check_prefix, var) and _prefix_filecheck_ir_name:
        var = _prefix_filecheck_ir_name + var
    var = var.replace('.', '_')
    var = var.replace('-', '_')
    return var.upper()

  # Create a FileCheck variable from regex.
  def get_value_definition(self, var, match):
    # for backwards compatibility we check locals with '.*'
    varname = self.get_value_name(var, self.check_prefix)
    prefix = self.get_ir_prefix_from_ir_value_match(match)[0]
    if self.is_number:
      regex = ''  # always capture a number in the default format
      capture_start = '[[#'
    else:
      regex = self.get_ir_regex_from_ir_value_re_match(match)
      capture_start = '[['
    if self.is_local_def_ir_value_match(match):
      return capture_start + varname + ':' + prefix + regex + ']]'
    return prefix + capture_start + varname + ':' + regex + ']]'

  # Use a FileCheck variable.
  def get_value_use(self, var, match, var_prefix=None):
    if var_prefix is None:
      var_prefix = self.check_prefix
    capture_start = '[[#' if self.is_number else '[['
    if self.is_local_def_ir_value_match(match):
      return capture_start + self.get_value_name(var, var_prefix) + ']]'
    prefix = self.get_ir_prefix_from_ir_value_match(match)[0]
    return prefix + capture_start + self.get_value_name(var, var_prefix) + ']]'

# Description of the different "unnamed" values we match in the IR, e.g.,
# (local) ssa values, (debug) metadata, etc.
ir_nameless_values = [
    NamelessValue(r'TMP'        , '%' , r'%'                   , None            , None                   , r'[\w$.-]+?' , None                 , False) ,
    NamelessValue(r'ATTR'       , '#' , r'#'                   , None            , None                   , r'[0-9]+'    , None                 , False) ,
    NamelessValue(r'ATTR'       , '#' , None                   , r'attributes #' , r'[0-9]+'              , None         , r'{[^}]*}'           , False) ,
    NamelessValue(r'GLOB'       , '@' , r'@'                   , None            , None                   , r'[0-9]+'    , None                 , False) ,
    NamelessValue(r'GLOB'       , '@' , None                   , r'@'            , r'[a-zA-Z0-9_$"\\.-]+' , None         , r'.+'                , True)  ,
    NamelessValue(r'DBG'        , '!' , r'!dbg '               , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'DIASSIGNID' , '!' , r'!DIAssignID '               , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'PROF'       , '!' , r'!prof '              , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'TBAA'       , '!' , r'!tbaa '              , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'TBAA_STRUCT', '!' , r'!tbaa.struct '       , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'RNG'        , '!' , r'!range '             , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'LOOP'       , '!' , r'!llvm.loop '         , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'META'       , '!' , r'metadata '           , None            , None                   , r'![0-9]+'   , None                 , False) ,
    NamelessValue(r'META'       , '!' , None                   , r''             , r'![0-9]+'             , None         , r'(?:distinct |)!.*' , False) ,
    NamelessValue(r'ACC_GRP'    , '!' , r'!llvm.access.group ' , None            , None                   , r'![0-9]+'   , None                 , False) ,
]

asm_nameless_values = [
 NamelessValue(r'MCINST', 'Inst#', None, '<MCInst #', r'\d+', None, r'.+',
               False, is_number=True, replace_number_with_counter=True),
 NamelessValue(r'MCREG',  'Reg:', None, '<MCOperand Reg:', r'\d+', None, r'.+',
               False, is_number=True, replace_number_with_counter=True),
]

def createOrRegexp(old, new):
  if not old:
    return new
  if not new:
    return old
  return old + '|' + new

def createPrefixMatch(prefix_str, prefix_re):
  if prefix_str is None or prefix_re is None:
    return ''
  return '(?:' + prefix_str + '(' + prefix_re + '))'

# Build the regexp that matches an "IR value". This can be a local variable,
# argument, global, or metadata, anything that is "named". It is important that
# the PREFIX and SUFFIX below only contain a single group, if that changes
# other locations will need adjustment as well.
IR_VALUE_REGEXP_PREFIX = r'(\s*)'
IR_VALUE_REGEXP_STRING = r''
for nameless_value in ir_nameless_values:
  lcl_match = createPrefixMatch(nameless_value.ir_prefix, nameless_value.ir_regexp)
  glb_match = createPrefixMatch(nameless_value.global_ir_prefix, nameless_value.global_ir_prefix_regexp)
  assert((lcl_match or glb_match) and not (lcl_match and glb_match))
  if lcl_match:
    IR_VALUE_REGEXP_STRING = createOrRegexp(IR_VALUE_REGEXP_STRING, lcl_match)
  elif glb_match:
    IR_VALUE_REGEXP_STRING = createOrRegexp(IR_VALUE_REGEXP_STRING, '^' + glb_match)
IR_VALUE_REGEXP_SUFFIX = r'([,\s\(\)]|\Z)'
IR_VALUE_RE = re.compile(IR_VALUE_REGEXP_PREFIX + r'(' + IR_VALUE_REGEXP_STRING + r')' + IR_VALUE_REGEXP_SUFFIX)

# Build the regexp that matches an "ASM value" (currently only for --asm-show-inst comments).
ASM_VALUE_REGEXP_STRING = ''
for nameless_value in asm_nameless_values:
  glb_match = createPrefixMatch(nameless_value.global_ir_prefix, nameless_value.global_ir_prefix_regexp)
  assert not nameless_value.ir_prefix and not nameless_value.ir_regexp
  ASM_VALUE_REGEXP_STRING = createOrRegexp(ASM_VALUE_REGEXP_STRING, glb_match)
ASM_VALUE_REGEXP_SUFFIX = r'([>\s]|\Z)'
ASM_VALUE_RE = re.compile(r'((?:#|//)\s*)' + '(' + ASM_VALUE_REGEXP_STRING + ')' + ASM_VALUE_REGEXP_SUFFIX)

# The entire match is group 0, the prefix has one group (=1), the entire
# IR_VALUE_REGEXP_STRING is one group (=2), and then the nameless values start.
first_nameless_group_in_ir_value_match = 3

# constants for the group id of special matches
variable_group_in_ir_value_match = 3
attribute_group_in_ir_value_match = 4

# Check a match for IR_VALUE_RE and inspect it to determine if it was a local
# value, %..., global @..., debug number !dbg !..., etc. See the PREFIXES above.
def get_idx_from_ir_value_match(match):
  for i in range(first_nameless_group_in_ir_value_match, match.lastindex):
    if match.group(i) is not None:
      return i - first_nameless_group_in_ir_value_match
  error("Unable to identify the kind of IR value from the match!")
  return 0

# See get_idx_from_ir_value_match
def get_name_from_ir_value_match(match):
  return match.group(get_idx_from_ir_value_match(match) + first_nameless_group_in_ir_value_match)

def get_nameless_value_from_match(match, nameless_values) -> NamelessValue:
  return nameless_values[get_idx_from_ir_value_match(match)]

# Return true if var clashes with the scripted FileCheck check_prefix.
def may_clash_with_default_check_prefix_name(check_prefix, var):
  return check_prefix and re.match(r'^' + check_prefix + r'[0-9]+?$', var, re.IGNORECASE)

def generalize_check_lines_common(lines, is_analyze, vars_seen,
                                  global_vars_seen, nameless_values,
                                  nameless_value_regex, is_asm):
  # This gets called for each match that occurs in
  # a line. We transform variables we haven't seen
  # into defs, and variables we have seen into uses.
  def transform_line_vars(match):
    var = get_name_from_ir_value_match(match)
    nameless_value = get_nameless_value_from_match(match, nameless_values)
    if may_clash_with_default_check_prefix_name(nameless_value.check_prefix, var):
      warn("Change IR value name '%s' or use --prefix-filecheck-ir-name to prevent possible conflict"
           " with scripted FileCheck name." % (var,))
    key = (var, nameless_value.check_key)
    is_local_def = nameless_value.is_local_def_ir_value_match(match)
    if is_local_def and key in vars_seen:
      rv = nameless_value.get_value_use(var, match)
    elif not is_local_def and key in global_vars_seen:
      # We could have seen a different prefix for the global variables first,
      # ensure we use that one instead of the prefix for the current match.
      rv = nameless_value.get_value_use(var, match, global_vars_seen[key])
    else:
      if is_local_def:
        vars_seen.add(key)
      else:
        global_vars_seen[key] = nameless_value.check_prefix
      rv = nameless_value.get_value_definition(var, match)
    # re.sub replaces the entire regex match
    # with whatever you return, so we have
    # to make sure to hand it back everything
    # including the commas and spaces.
    return match.group(1) + rv + match.group(match.lastindex)

  lines_with_def = []

  for i, line in enumerate(lines):
    if not is_asm:
      # An IR variable named '%.' matches the FileCheck regex string.
      line = line.replace('%.', '%dot')
      for regex in _global_hex_value_regex:
        if re.match('^@' + regex + ' = ', line):
          line = re.sub(r'\bi([0-9]+) ([0-9]+)',
              lambda m : 'i' + m.group(1) + ' [[#' + hex(int(m.group(2))) + ']]',
              line)
          break
      # Ignore any comments, since the check lines will too.
      scrubbed_line = SCRUB_IR_COMMENT_RE.sub(r'', line)
      lines[i] = scrubbed_line
    if is_asm or not is_analyze:
      # It can happen that two matches are back-to-back and for some reason sub
      # will not replace both of them. For now we work around this by
      # substituting until there is no more match.
      changed = True
      while changed:
        (lines[i], changed) = nameless_value_regex.subn(transform_line_vars,
                                                        lines[i], count=1)
  return lines

# Replace IR value defs and uses with FileCheck variables.
def generalize_check_lines(lines, is_analyze, vars_seen, global_vars_seen):
  return generalize_check_lines_common(lines, is_analyze, vars_seen,
                                       global_vars_seen, ir_nameless_values,
                                       IR_VALUE_RE, False)

def generalize_asm_check_lines(lines, vars_seen, global_vars_seen):
  return generalize_check_lines_common(lines, False, vars_seen,
                                       global_vars_seen, asm_nameless_values,
                                       ASM_VALUE_RE, True)

def add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name, check_label_format, is_backend, is_analyze, global_vars_seen_dict, is_filtered):
  # prefix_exclusions are prefixes we cannot use to print the function because it doesn't exist in run lines that use these prefixes as well.
  prefix_exclusions = set()
  printed_prefixes = []
  for p in prefix_list:
    checkprefixes = p[0]
    # If not all checkprefixes of this run line produced the function we cannot check for it as it does not
    # exist for this run line. A subset of the check prefixes might know about the function but only because
    # other run lines created it.
    if any(map(lambda checkprefix: func_name not in func_dict[checkprefix], checkprefixes)):
      prefix_exclusions |= set(checkprefixes)
      continue

  # prefix_exclusions is constructed, we can now emit the output
  for p in prefix_list:
    global_vars_seen = {}
    checkprefixes = p[0]
    for checkprefix in checkprefixes:
      if checkprefix in global_vars_seen_dict:
        global_vars_seen.update(global_vars_seen_dict[checkprefix])
      else:
        global_vars_seen_dict[checkprefix] = {}
      if checkprefix in printed_prefixes:
        break

      # Check if the prefix is excluded.
      if checkprefix in prefix_exclusions:
        continue

      # If we do not have output for this prefix we skip it.
      if not func_dict[checkprefix][func_name]:
        continue

      # Add some space between different check prefixes, but not after the last
      # check line (before the test code).
      if is_backend:
        if len(printed_prefixes) != 0:
          output_lines.append(comment_marker)

      if checkprefix not in global_vars_seen_dict:
        global_vars_seen_dict[checkprefix] = {}

      global_vars_seen_before = [key for key in global_vars_seen.keys()]

      vars_seen = set()
      printed_prefixes.append(checkprefix)
      attrs = str(func_dict[checkprefix][func_name].attrs)
      attrs = '' if attrs == 'None' else attrs
      if attrs:
        output_lines.append('%s %s: Function Attrs: %s' % (comment_marker, checkprefix, attrs))
      args_and_sig = str(func_dict[checkprefix][func_name].args_and_sig)
      if args_and_sig:
        args_and_sig = generalize_check_lines([args_and_sig], is_analyze, vars_seen, global_vars_seen)[0]
      func_name_separator = func_dict[checkprefix][func_name].func_name_separator
      if '[[' in args_and_sig:
        output_lines.append(check_label_format % (checkprefix, func_name, '', func_name_separator))
        output_lines.append('%s %s-SAME: %s' % (comment_marker, checkprefix, args_and_sig))
      else:
        output_lines.append(check_label_format % (checkprefix, func_name, args_and_sig, func_name_separator))
      func_body = str(func_dict[checkprefix][func_name]).splitlines()
      if not func_body:
        # We have filtered everything.
        continue

      # For ASM output, just emit the check lines.
      if is_backend:
        body_start = 1
        if is_filtered:
          # For filtered output we don't add "-NEXT" so don't add extra spaces
          # before the first line.
          body_start = 0
        else:
          output_lines.append('%s %s:       %s' % (comment_marker, checkprefix, func_body[0]))
        func_lines = generalize_asm_check_lines(func_body[body_start:],
                                                vars_seen, global_vars_seen)
        for func_line in func_lines:
          if func_line.strip() == '':
            output_lines.append('%s %s-EMPTY:' % (comment_marker, checkprefix))
          else:
            check_suffix = '-NEXT' if not is_filtered else ''
            output_lines.append('%s %s%s:  %s' % (comment_marker, checkprefix,
                                                  check_suffix, func_line))
        # Remember new global variables we have not seen before
        for key in global_vars_seen:
          if key not in global_vars_seen_before:
            global_vars_seen_dict[checkprefix][key] = global_vars_seen[key]
        break

      # For IR output, change all defs to FileCheck variables, so we're immune
      # to variable naming fashions.
      func_body = generalize_check_lines(func_body, is_analyze, vars_seen, global_vars_seen)

      # This could be selectively enabled with an optional invocation argument.
      # Disabled for now: better to check everything. Be safe rather than sorry.

      # Handle the first line of the function body as a special case because
      # it's often just noise (a useless asm comment or entry label).
      #if func_body[0].startswith("#") or func_body[0].startswith("entry:"):
      #  is_blank_line = True
      #else:
      #  output_lines.append('%s %s:       %s' % (comment_marker, checkprefix, func_body[0]))
      #  is_blank_line = False

      is_blank_line = False

      for func_line in func_body:
        if func_line.strip() == '':
          is_blank_line = True
          continue
        # Do not waste time checking IR comments.
        func_line = SCRUB_IR_COMMENT_RE.sub(r'', func_line)

        # Skip blank lines instead of checking them.
        if is_blank_line:
          output_lines.append('{} {}:       {}'.format(
              comment_marker, checkprefix, func_line))
        else:
          check_suffix = '-NEXT' if not is_filtered else ''
          output_lines.append('{} {}{}:  {}'.format(
              comment_marker, checkprefix, check_suffix, func_line))
        is_blank_line = False

      # Add space between different check prefixes and also before the first
      # line of code in the test function.
      output_lines.append(comment_marker)

      # Remember new global variables we have not seen before
      for key in global_vars_seen:
        if key not in global_vars_seen_before:
          global_vars_seen_dict[checkprefix][key] = global_vars_seen[key]
      break
  return printed_prefixes

def add_ir_checks(output_lines, comment_marker, prefix_list, func_dict,
                  func_name, preserve_names, function_sig,
                  global_vars_seen_dict, is_filtered):
  # Label format is based on IR string.
  function_def_regex = 'define {{[^@]+}}' if function_sig else ''
  check_label_format = '{} %s-LABEL: {}@%s%s%s'.format(comment_marker, function_def_regex)
  return add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name,
                    check_label_format, False, preserve_names, global_vars_seen_dict,
                    is_filtered)

def add_analyze_checks(output_lines, comment_marker, prefix_list, func_dict, func_name, is_filtered):
  check_label_format = '{} %s-LABEL: \'%s%s%s\''.format(comment_marker)
  global_vars_seen_dict = {}
  return add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name,
                    check_label_format, False, True, global_vars_seen_dict,
                    is_filtered)

def build_global_values_dictionary(glob_val_dict, raw_tool_output, prefixes):
  for nameless_value in itertools.chain(ir_nameless_values, asm_nameless_values):
    if nameless_value.global_ir_prefix is None:
      continue

    lhs_re_str = nameless_value.global_ir_prefix + nameless_value.global_ir_prefix_regexp
    rhs_re_str = nameless_value.global_ir_rhs_regexp

    global_ir_value_re_str = r'^' + lhs_re_str + r'\s=\s' + rhs_re_str + r'$'
    global_ir_value_re = re.compile(global_ir_value_re_str, flags=(re.M))
    lines = []
    for m in global_ir_value_re.finditer(raw_tool_output):
      lines.append(m.group(0))

    for prefix in prefixes:
      if glob_val_dict[prefix] is None:
        continue
      if nameless_value.check_prefix in glob_val_dict[prefix]:
        if lines == glob_val_dict[prefix][nameless_value.check_prefix]:
          continue
        if prefix == prefixes[-1]:
          warn('Found conflicting asm under the same prefix: %r!' % (prefix,))
        else:
          glob_val_dict[prefix][nameless_value.check_prefix] = None
          continue
      glob_val_dict[prefix][nameless_value.check_prefix] = lines

def add_global_checks(glob_val_dict, comment_marker, prefix_list, output_lines, global_vars_seen_dict, is_analyze, is_before_functions):
  printed_prefixes = set()
  for nameless_value in ir_nameless_values:
    if nameless_value.global_ir_prefix is None:
      continue
    if nameless_value.is_before_functions != is_before_functions:
      continue
    for p in prefix_list:
      global_vars_seen = {}
      checkprefixes = p[0]
      if checkprefixes is None:
        continue
      for checkprefix in checkprefixes:
        if checkprefix in global_vars_seen_dict:
          global_vars_seen.update(global_vars_seen_dict[checkprefix])
        else:
          global_vars_seen_dict[checkprefix] = {}
        if (checkprefix, nameless_value.check_prefix) in printed_prefixes:
          break
        if not glob_val_dict[checkprefix]:
          continue
        if nameless_value.check_prefix not in glob_val_dict[checkprefix]:
          continue
        if not glob_val_dict[checkprefix][nameless_value.check_prefix]:
          continue

        check_lines = []
        global_vars_seen_before = [key for key in global_vars_seen.keys()]
        for line in glob_val_dict[checkprefix][nameless_value.check_prefix]:
          if _global_value_regex:
            matched = False
            for regex in _global_value_regex:
              if re.match('^@' + regex + ' = ', line):
                matched = True
                break
            if not matched:
              continue
          tmp = generalize_check_lines([line], is_analyze, set(), global_vars_seen)
          check_line = '%s %s: %s' % (comment_marker, checkprefix, tmp[0])
          check_lines.append(check_line)
        if not check_lines:
          continue

        output_lines.append(comment_marker + SEPARATOR)
        for check_line in check_lines:
          output_lines.append(check_line)

        printed_prefixes.add((checkprefix, nameless_value.check_prefix))

        # Remembe new global variables we have not seen before
        for key in global_vars_seen:
          if key not in global_vars_seen_before:
            global_vars_seen_dict[checkprefix][key] = global_vars_seen[key]
        break

  if printed_prefixes:
    output_lines.append(comment_marker + SEPARATOR)
  return printed_prefixes


def check_prefix(prefix):
  if not PREFIX_RE.match(prefix):
    hint = ""
    if ',' in prefix:
      hint = " Did you mean '--check-prefixes=" + prefix + "'?"
    warn(("Supplied prefix '%s' is invalid. Prefix must contain only alphanumeric characters, hyphens and underscores." + hint) %
         (prefix))


def verify_filecheck_prefixes(fc_cmd):
  fc_cmd_parts = fc_cmd.split()
  for part in fc_cmd_parts:
    if "check-prefix=" in part:
      prefix = part.split('=', 1)[1]
      check_prefix(prefix)
    elif "check-prefixes=" in part:
      prefixes = part.split('=', 1)[1].split(',')
      for prefix in prefixes:
        check_prefix(prefix)
        if prefixes.count(prefix) > 1:
          warn("Supplied prefix '%s' is not unique in the prefix list." % (prefix,))


def get_autogennote_suffix(parser, args):
  autogenerated_note_args = ''
  for action in parser._actions:
    if not hasattr(args, action.dest):
      continue  # Ignore options such as --help that aren't included in args
    # Ignore parameters such as paths to the binary or the list of tests
    if action.dest in ('tests', 'update_only', 'tool_binary', 'opt_binary',
                       'llc_binary', 'clang', 'opt', 'llvm_bin', 'verbose',
                       'force_update'):
      continue
    value = getattr(args, action.dest)
    if action.const is not None:  # action stores a constant (usually True/False)
      # Skip actions with different constant values (this happens with boolean
      # --foo/--no-foo options)
      if value != action.const:
        continue
    if parser.get_default(action.dest) == value:
      continue  # Don't add default values
    if action.dest == 'filters':
      # Create a separate option for each filter element.  The value is a list
      # of Filter objects.
      for elem in value:
        opt_name = 'filter-out' if elem.is_filter_out else 'filter'
        opt_value = elem.pattern()
        new_arg = '--%s "%s" ' % (opt_name, opt_value.strip('"'))
        if new_arg not in autogenerated_note_args:
          autogenerated_note_args += new_arg
    else:
      autogenerated_note_args += action.option_strings[0] + ' '
      if action.const is None:  # action takes a parameter
        if action.nargs == '+':
          value = ' '.join(map(lambda v: '"' + v.strip('"') + '"', value))
        autogenerated_note_args += '%s ' % value
  if autogenerated_note_args:
    autogenerated_note_args = ' %s %s' % (UTC_ARGS_KEY, autogenerated_note_args[:-1])
  return autogenerated_note_args


def check_for_command(line, parser, args, argv, argparse_callback):
  cmd_m = UTC_ARGS_CMD.match(line)
  if cmd_m:
    for option in shlex.split(cmd_m.group('cmd').strip()):
      if option:
        argv.append(option)
    args = parser.parse_args(filter(lambda arg: arg not in args.tests, argv))
    if argparse_callback is not None:
      argparse_callback(args)
  return args, argv

def find_arg_in_test(test_info, get_arg_to_check, arg_string, is_global):
  result = get_arg_to_check(test_info.args)
  if not result and is_global:
    # See if this has been specified via UTC_ARGS.  This is a "global" option
    # that affects the entire generation of test checks.  If it exists anywhere
    # in the test, apply it to everything.
    saw_line = False
    for line_info in test_info.ro_iterlines():
      line = line_info.line
      if not line.startswith(';') and line.strip() != '':
        saw_line = True
      result = get_arg_to_check(line_info.args)
      if result:
        if warn and saw_line:
          # We saw the option after already reading some test input lines.
          # Warn about it.
          print('WARNING: Found {} in line following test start: '.format(arg_string)
                + line, file=sys.stderr)
          print('WARNING: Consider moving {} to top of file'.format(arg_string),
                file=sys.stderr)
        break
  return result

def dump_input_lines(output_lines, test_info, prefix_set, comment_string):
  for input_line_info in test_info.iterlines(output_lines):
    line = input_line_info.line
    args = input_line_info.args
    if line.strip() == comment_string:
      continue
    if line.strip() == comment_string + SEPARATOR:
      continue
    if line.lstrip().startswith(comment_string):
      m = CHECK_RE.match(line)
      if m and m.group(1) in prefix_set:
        continue
    output_lines.append(line.rstrip('\n'))

def add_checks_at_end(output_lines, prefix_list, func_order,
                      comment_string, check_generator):
  added = set()
  generated_prefixes = set()
  for prefix in prefix_list:
    prefixes = prefix[0]
    tool_args = prefix[1]
    for prefix in prefixes:
      for func in func_order[prefix]:
        # The func order can contain the same functions multiple times.
        # If we see one again we are done.
        if (func, prefix) in added:
          continue
        if added:
          output_lines.append(comment_string)

        # The add_*_checks routines expect a run list whose items are
        # tuples that have a list of prefixes as their first element and
        # tool command args string as their second element.  They output
        # checks for each prefix in the list of prefixes.  By doing so, it
        # implicitly assumes that for each function every run line will
        # generate something for that function.  That is not the case for
        # generated functions as some run lines might not generate them
        # (e.g. -fopenmp vs. no -fopenmp).
        #
        # Therefore, pass just the prefix we're interested in.  This has
        # the effect of generating all of the checks for functions of a
        # single prefix before moving on to the next prefix.  So checks
        # are ordered by prefix instead of by function as in "normal"
        # mode.
        for generated_prefix in check_generator(output_lines,
                        [([prefix], tool_args)], func):
          added.add((func, generated_prefix))
          generated_prefixes.add(generated_prefix)
  return generated_prefixes
