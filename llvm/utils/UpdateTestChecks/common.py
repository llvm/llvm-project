import re
import subprocess

RUN_LINE_RE = re.compile('^\s*;\s*RUN:\s*(.*)$')
CHECK_PREFIX_RE = re.compile('--?check-prefix(?:es)?=(\S+)')
CHECK_RE = re.compile(r'^\s*;\s*([^:]+?)(?:-NEXT|-NOT|-DAG|-LABEL)?:')

IR_FUNCTION_RE = re.compile('^\s*define\s+(?:internal\s+)?[^@]*@(\w+)\s*\(')
TRIPLE_IR_RE = re.compile(r'^target\s+triple\s*=\s*"([^"]+)"$')
TRIPLE_ARG_RE = re.compile(r'-mtriple=([^ ]+)')

SCRUB_LEADING_WHITESPACE_RE = re.compile(r'^(\s+)')
SCRUB_WHITESPACE_RE = re.compile(r'(?!^(|  \w))[ \t]+', flags=re.M)
SCRUB_TRAILING_WHITESPACE_RE = re.compile(r'[ \t]+$', flags=re.M)
SCRUB_KILL_COMMENT_RE = re.compile(r'^ *#+ +kill:.*\n')
SCRUB_LOOP_COMMENT_RE = re.compile(
    r'# =>This Inner Loop Header:.*|# in Loop:.*', flags=re.M)

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
    stdout = subprocess.check_output(exe + ' ' + cmd_args,
                                     shell=True, stdin=ir_file)
  # Fix line endings to unix CR style.
  stdout = stdout.replace('\r\n', '\n')
  return stdout

# Build up a dictionary of all the function bodies.
def build_function_body_dictionary(function_re, scrubber, scrubber_args, raw_tool_output, prefixes, func_dict, verbose):
  for m in function_re.finditer(raw_tool_output):
    if not m:
      continue
    func = m.group('func')
    scrubbed_body = scrubber(m.group('body'), *scrubber_args)
    if func.startswith('stress'):
      # We only use the last line of the function body for stress tests.
      scrubbed_body = '\n'.join(scrubbed_body.splitlines()[-1:])
    if verbose:
      print >>sys.stderr, 'Processing function: ' + func
      for l in scrubbed_body.splitlines():
        print >>sys.stderr, '  ' + l
    for prefix in prefixes:
      if func in func_dict[prefix] and func_dict[prefix][func] != scrubbed_body:
        if prefix == prefixes[-1]:
          print >>sys.stderr, ('WARNING: Found conflicting asm under the '
                               'same prefix: %r!' % (prefix,))
        else:
          func_dict[prefix][func] = None
          continue

      func_dict[prefix][func] = scrubbed_body
