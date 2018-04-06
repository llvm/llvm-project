from __future__ import print_function
import re
import string
import subprocess
import sys

if sys.version_info[0] > 2:
  class string:
    expandtabs = str.expandtabs
else:
  import string

##### Common utilities for update_*test_checks.py

def should_add_line_to_output(input_line, prefix_set):
  # Skip any blank comment lines in the IR.
  if input_line.strip() == ';':
    return False
  # Skip any blank lines in the IR.
  #if input_line.strip() == '':
  #  return False
  # And skip any CHECK lines. We're building our own.
  m = CHECK_RE.match(input_line)
  if m and m.group(1) in prefix_set:
    return False

  return True

# Invoke the tool that is being tested.
def invoke_tool(exe, cmd_args, ir):
  with open(ir) as ir_file:
    # TODO Remove the str form which is used by update_test_checks.py and
    # update_llc_test_checks.py
    # The safer list form is used by update_cc_test_checks.py
    if isinstance(cmd_args, list):
      stdout = subprocess.check_output([exe] + cmd_args, stdin=ir_file)
    else:
      stdout = subprocess.check_output(exe + ' ' + cmd_args,
                                       shell=True, stdin=ir_file)
    if sys.version_info[0] > 2:
      stdout = stdout.decode()
  # Fix line endings to unix CR style.
  return stdout.replace('\r\n', '\n')

##### LLVM IR parser

RUN_LINE_RE = re.compile('^\s*[;#]\s*RUN:\s*(.*)$')
CHECK_PREFIX_RE = re.compile('--?check-prefix(?:es)?[= ](\S+)')
CHECK_RE = re.compile(r'^\s*[;#]\s*([^:]+?)(?:-NEXT|-NOT|-DAG|-LABEL)?:')

OPT_FUNCTION_RE = re.compile(
    r'^\s*define\s+(?:internal\s+)?[^@]*@(?P<func>[\w-]+?)\s*\('
    r'(\s+)?[^)]*[^{]*\{\n(?P<body>.*?)^\}$',
    flags=(re.M | re.S))

ANALYZE_FUNCTION_RE = re.compile(
    r'^\s*\'(?P<analysis>[\w\s-]+?)\'\s+for\s+function\s+\'(?P<func>[\w-]+?)\':'
    r'\s*\n(?P<body>.*)$',
    flags=(re.X | re.S))

IR_FUNCTION_RE = re.compile('^\s*define\s+(?:internal\s+)?[^@]*@(\w+)\s*\(')
TRIPLE_IR_RE = re.compile(r'^\s*target\s+triple\s*=\s*"([^"]+)"$')
TRIPLE_ARG_RE = re.compile(r'-mtriple[= ]([^ ]+)')
MARCH_ARG_RE = re.compile(r'-march[= ]([^ ]+)')

SCRUB_LEADING_WHITESPACE_RE = re.compile(r'^(\s+)')
SCRUB_WHITESPACE_RE = re.compile(r'(?!^(|  \w))[ \t]+', flags=re.M)
SCRUB_TRAILING_WHITESPACE_RE = re.compile(r'[ \t]+$', flags=re.M)
SCRUB_KILL_COMMENT_RE = re.compile(r'^ *#+ +kill:.*\n')
SCRUB_LOOP_COMMENT_RE = re.compile(
    r'# =>This Inner Loop Header:.*|# in Loop:.*', flags=re.M)

def scrub_body(body):
  # Scrub runs of whitespace out of the assembly, but leave the leading
  # whitespace in place.
  body = SCRUB_WHITESPACE_RE.sub(r' ', body)
  # Expand the tabs used for indentation.
  body = string.expandtabs(body, 2)
  # Strip trailing whitespace.
  body = SCRUB_TRAILING_WHITESPACE_RE.sub(r'', body)
  return body

# Build up a dictionary of all the function bodies.
def build_function_body_dictionary(function_re, scrubber, scrubber_args, raw_tool_output, prefixes, func_dict, verbose):
  for m in function_re.finditer(raw_tool_output):
    if not m:
      continue
    func = m.group('func')
    scrubbed_body = scrubber(m.group('body'), *scrubber_args)
    if m.groupdict().has_key('analysis'):
      analysis = m.group('analysis')
      if analysis.lower() != 'cost model analysis':
        print('WARNING: Unsupported analysis mode: %r!' % (analysis,), file=sys.stderr)
    if func.startswith('stress'):
      # We only use the last line of the function body for stress tests.
      scrubbed_body = '\n'.join(scrubbed_body.splitlines()[-1:])
    if verbose:
      print('Processing function: ' + func, file=sys.stderr)
      for l in scrubbed_body.splitlines():
        print('  ' + l, file=sys.stderr)
    for prefix in prefixes:
      if func in func_dict[prefix] and func_dict[prefix][func] != scrubbed_body:
        if prefix == prefixes[-1]:
          print('WARNING: Found conflicting asm under the '
                               'same prefix: %r!' % (prefix,), file=sys.stderr)
        else:
          func_dict[prefix][func] = None
          continue

      func_dict[prefix][func] = scrubbed_body

##### Generator of LLVM IR CHECK lines

SCRUB_IR_COMMENT_RE = re.compile(r'\s*;.*')

# Match things that look at identifiers, but only if they are followed by
# spaces, commas, paren, or end of the string
IR_VALUE_RE = re.compile(r'(\s+)%([\w\.\-]+?)([,\s\(\)]|\Z)')

# Create a FileCheck variable name based on an IR name.
def get_value_name(var):
  if var.isdigit():
    var = 'TMP' + var
  var = var.replace('.', '_')
  var = var.replace('-', '_')
  return var.upper()


# Create a FileCheck variable from regex.
def get_value_definition(var):
  return '[[' + get_value_name(var) + ':%.*]]'


# Use a FileCheck variable.
def get_value_use(var):
  return '[[' + get_value_name(var) + ']]'

# Replace IR value defs and uses with FileCheck variables.
def genericize_check_lines(lines, is_analyze):
  # This gets called for each match that occurs in
  # a line. We transform variables we haven't seen
  # into defs, and variables we have seen into uses.
  def transform_line_vars(match):
    var = match.group(2)
    if var in vars_seen:
      rv = get_value_use(var)
    else:
      vars_seen.add(var)
      rv = get_value_definition(var)
    # re.sub replaces the entire regex match
    # with whatever you return, so we have
    # to make sure to hand it back everything
    # including the commas and spaces.
    return match.group(1) + rv + match.group(3)

  vars_seen = set()
  lines_with_def = []

  for i, line in enumerate(lines):
    # An IR variable named '%.' matches the FileCheck regex string.
    line = line.replace('%.', '%dot')
    # Ignore any comments, since the check lines will too.
    scrubbed_line = SCRUB_IR_COMMENT_RE.sub(r'', line)
    if is_analyze == False:
      lines[i] =  IR_VALUE_RE.sub(transform_line_vars, scrubbed_line)
    else:
      lines[i] =  scrubbed_line
  return lines


def add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name, check_label_format, is_asm, is_analyze):
  printed_prefixes = []
  for p in prefix_list:
    checkprefixes = p[0]
    for checkprefix in checkprefixes:
      if checkprefix in printed_prefixes:
        break
      # TODO func_dict[checkprefix] may be None, '' or not exist.
      # Fix the call sites.
      if func_name not in func_dict[checkprefix] or not func_dict[checkprefix][func_name]:
        continue

      # Add some space between different check prefixes, but not after the last
      # check line (before the test code).
      if is_asm == True:
        if len(printed_prefixes) != 0:
          output_lines.append(comment_marker)

      printed_prefixes.append(checkprefix)
      output_lines.append(check_label_format % (checkprefix, func_name))
      func_body = func_dict[checkprefix][func_name].splitlines()

      # For ASM output, just emit the check lines.
      if is_asm == True:
        output_lines.append('%s %s:       %s' % (comment_marker, checkprefix, func_body[0]))
        for func_line in func_body[1:]:
          output_lines.append('%s %s-NEXT:  %s' % (comment_marker, checkprefix, func_line))
        break

      # For IR output, change all defs to FileCheck variables, so we're immune
      # to variable naming fashions.
      func_body = genericize_check_lines(func_body, is_analyze)

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
        if is_blank_line == True:
          output_lines.append('{} {}:       {}'.format(
              comment_marker, checkprefix, func_line))
        else:
          output_lines.append('{} {}-NEXT:  {}'.format(
              comment_marker, checkprefix, func_line))
        is_blank_line = False

      # Add space between different check prefixes and also before the first
      # line of code in the test function.
      output_lines.append(comment_marker)
      break

def add_ir_checks(output_lines, comment_marker, prefix_list, func_dict, func_name):
  # Label format is based on IR string.
  check_label_format = '{} %s-LABEL: @%s('.format(comment_marker)
  add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name, check_label_format, False, False)

def add_analyze_checks(output_lines, comment_marker, prefix_list, func_dict, func_name):
  check_label_format = '{} %s-LABEL: \'%s\''.format(comment_marker)
  add_checks(output_lines, comment_marker, prefix_list, func_dict, func_name, check_label_format, False, True)
