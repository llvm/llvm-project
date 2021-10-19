#!/usr/bin/env python3

"""Updates FileCheck checks in MIR tests.

This script is a utility to update MIR based tests with new FileCheck
patterns.

The checks added by this script will cover the entire body of each
function it handles. Virtual registers used are given names via
FileCheck patterns, so if you do want to check a subset of the body it
should be straightforward to trim out the irrelevant parts. None of
the YAML metadata will be checked, other than function names.

If there are multiple llc commands in a test, the full set of checks
will be repeated for each different check pattern. Checks for patterns
that are common between different commands will be left as-is by
default, or removed if the --remove-common-prefixes flag is provided.
"""

from __future__ import print_function

import argparse
import collections
import glob
import os
import re
import subprocess
import sys

from UpdateTestChecks import common

MIR_FUNC_NAME_RE = re.compile(r' *name: *(?P<func>[A-Za-z0-9_.-]+)')
MIR_BODY_BEGIN_RE = re.compile(r' *body: *\|')
MIR_BASIC_BLOCK_RE = re.compile(r' *bb\.[0-9]+.*:$')
VREG_RE = re.compile(r'(%[0-9]+)(?::[a-z0-9_]+)?(?:\([<>a-z0-9 ]+\))?')
MI_FLAGS_STR= (
    r'(frame-setup |frame-destroy |nnan |ninf |nsz |arcp |contract |afn '
    r'|reassoc |nuw |nsw |exact |fpexcept )*')
VREG_DEF_RE = re.compile(
    r'^ *(?P<vregs>{0}(?:, {0})*) = '
    r'{1}(?P<opcode>[A-Zt][A-Za-z0-9_]+)'.format(VREG_RE.pattern, MI_FLAGS_STR))
MIR_PREFIX_DATA_RE = re.compile(r'^ *(;|bb.[0-9].*: *$|[a-z]+:( |$)|$)')

IR_FUNC_NAME_RE = re.compile(
    r'^\s*define\s+(?:internal\s+)?[^@]*@(?P<func>[A-Za-z0-9_.]+)\s*\(')
IR_PREFIX_DATA_RE = re.compile(r'^ *(;|$)')

MIR_FUNC_RE = re.compile(
    r'^---$'
    r'\n'
    r'^ *name: *(?P<func>[A-Za-z0-9_.-]+)$'
    r'.*?'
    r'^ *body: *\|\n'
    r'(?P<body>.*?)\n'
    r'^\.\.\.$',
    flags=(re.M | re.S))


class LLC:
    def __init__(self, bin):
        self.bin = bin

    def __call__(self, args, ir):
        if ir.endswith('.mir'):
            args = '{} -x mir'.format(args)
        with open(ir) as ir_file:
            stdout = subprocess.check_output('{} {}'.format(self.bin, args),
                                             shell=True, stdin=ir_file)
            if sys.version_info[0] > 2:
              stdout = stdout.decode()
            # Fix line endings to unix CR style.
            stdout = stdout.replace('\r\n', '\n')
        return stdout


class Run:
    def __init__(self, prefixes, cmd_args, triple):
        self.prefixes = prefixes
        self.cmd_args = cmd_args
        self.triple = triple

    def __getitem__(self, index):
        return [self.prefixes, self.cmd_args, self.triple][index]


def log(msg, verbose=True):
    if verbose:
        print(msg, file=sys.stderr)


def find_triple_in_ir(lines, verbose=False):
    for l in lines:
        m = common.TRIPLE_IR_RE.match(l)
        if m:
            return m.group(1)
    return None


def build_run_list(test, run_lines, verbose=False):
    run_list = []
    all_prefixes = []
    for l in run_lines:
        if '|' not in l:
            common.warn('Skipping unparseable RUN line: ' + l)
            continue

        commands = [cmd.strip() for cmd in l.split('|', 1)]
        llc_cmd = commands[0]
        filecheck_cmd = commands[1] if len(commands) > 1 else ''
        common.verify_filecheck_prefixes(filecheck_cmd)

        if not llc_cmd.startswith('llc '):
            common.warn('Skipping non-llc RUN line: {}'.format(l), test_file=test)
            continue
        if not filecheck_cmd.startswith('FileCheck '):
            common.warn('Skipping non-FileChecked RUN line: {}'.format(l),
                 test_file=test)
            continue

        triple = None
        m = common.TRIPLE_ARG_RE.search(llc_cmd)
        if m:
            triple = m.group(1)
        # If we find -march but not -mtriple, use that.
        m = common.MARCH_ARG_RE.search(llc_cmd)
        if m and not triple:
            triple = '{}--'.format(m.group(1))

        cmd_args = llc_cmd[len('llc'):].strip()
        cmd_args = cmd_args.replace('< %s', '').replace('%s', '').strip()

        check_prefixes = [
            item
            for m in common.CHECK_PREFIX_RE.finditer(filecheck_cmd)
            for item in m.group(1).split(',')]
        if not check_prefixes:
            check_prefixes = ['CHECK']
        all_prefixes += check_prefixes

        run_list.append(Run(check_prefixes, cmd_args, triple))

    # Remove any common prefixes. We'll just leave those entirely alone.
    common_prefixes = set([prefix for prefix in all_prefixes
                           if all_prefixes.count(prefix) > 1])
    for run in run_list:
        run.prefixes = [p for p in run.prefixes if p not in common_prefixes]

    return run_list, common_prefixes


def find_functions_with_one_bb(lines, verbose=False):
    result = []
    cur_func = None
    bbs = 0
    for line in lines:
        m = MIR_FUNC_NAME_RE.match(line)
        if m:
            if bbs == 1:
                result.append(cur_func)
            cur_func = m.group('func')
            bbs = 0
        m = MIR_BASIC_BLOCK_RE.match(line)
        if m:
            bbs += 1
    if bbs == 1:
        result.append(cur_func)
    return result


def build_function_body_dictionary(test, raw_tool_output, triple, prefixes,
                                   func_dict, verbose):
    for m in MIR_FUNC_RE.finditer(raw_tool_output):
        func = m.group('func')
        body = m.group('body')
        if verbose:
            log('Processing function: {}'.format(func))
            for l in body.splitlines():
                log('  {}'.format(l))
        for prefix in prefixes:
            if func in func_dict[prefix] and func_dict[prefix][func] != body:
                common.warn('Found conflicting asm for prefix: {}'.format(prefix),
                     test_file=test)
            func_dict[prefix][func] = body


def add_checks_for_function(test, output_lines, run_list, func_dict, func_name,
                            single_bb, verbose=False):
    printed_prefixes = set()
    for run in run_list:
        for prefix in run.prefixes:
            if prefix in printed_prefixes:
                continue
            if not func_dict[prefix][func_name]:
                continue
            # if printed_prefixes:
            #     # Add some space between different check prefixes.
            #     output_lines.append('')
            printed_prefixes.add(prefix)
            log('Adding {} lines for {}'.format(prefix, func_name), verbose)
            add_check_lines(test, output_lines, prefix, func_name, single_bb,
                            func_dict[prefix][func_name].splitlines())
            break
    return output_lines


def add_check_lines(test, output_lines, prefix, func_name, single_bb,
                    func_body):
    if single_bb:
        # Don't bother checking the basic block label for a single BB
        func_body.pop(0)

    if not func_body:
        common.warn('Function has no instructions to check: {}'.format(func_name),
             test_file=test)
        return

    first_line = func_body[0]
    indent = len(first_line) - len(first_line.lstrip(' '))
    # A check comment, indented the appropriate amount
    check = '{:>{}}; {}'.format('', indent, prefix)

    output_lines.append('{}-LABEL: name: {}'.format(check, func_name))
    first_check = True

    vreg_map = {}
    for func_line in func_body:
        if not func_line.strip():
            # The mir printer prints leading whitespace so we can't use CHECK-EMPTY:
            output_lines.append(check + '-NEXT: {{' + func_line + '$}}')
            continue
        m = VREG_DEF_RE.match(func_line)
        if m:
            for vreg in VREG_RE.finditer(m.group('vregs')):
                name = mangle_vreg(m.group('opcode'), vreg_map.values())
                vreg_map[vreg.group(1)] = name
                func_line = func_line.replace(
                    vreg.group(1), '[[{}:%[0-9]+]]'.format(name), 1)
        for number, name in vreg_map.items():
            func_line = re.sub(r'{}\b'.format(number), '[[{}]]'.format(name),
                               func_line)
        filecheck_directive = check if first_check else check + '-NEXT'
        first_check = False
        check_line = '{}: {}'.format(filecheck_directive, func_line[indent:]).rstrip()
        output_lines.append(check_line)


def mangle_vreg(opcode, current_names):
    base = opcode
    # Simplify some common prefixes and suffixes
    if opcode.startswith('G_'):
        base = base[len('G_'):]
    if opcode.endswith('_PSEUDO'):
        base = base[:len('_PSEUDO')]
    # Shorten some common opcodes with long-ish names
    base = dict(IMPLICIT_DEF='DEF',
                GLOBAL_VALUE='GV',
                CONSTANT='C',
                FCONSTANT='C',
                MERGE_VALUES='MV',
                UNMERGE_VALUES='UV',
                INTRINSIC='INT',
                INTRINSIC_W_SIDE_EFFECTS='INT',
                INSERT_VECTOR_ELT='IVEC',
                EXTRACT_VECTOR_ELT='EVEC',
                SHUFFLE_VECTOR='SHUF').get(base, base)
    # Avoid ambiguity when opcodes end in numbers
    if len(base.rstrip('0123456789')) < len(base):
        base += '_'

    i = 0
    for name in current_names:
        if name.rstrip('0123456789') == base:
            i += 1
    if i:
        return '{}{}'.format(base, i)
    return base


def should_add_line_to_output(input_line, prefix_set):
    # Skip any check lines that we're handling.
    m = common.CHECK_RE.match(input_line)
    if m and m.group(1) in prefix_set:
        return False
    return True


def update_test_file(args, test):
    with open(test) as fd:
        input_lines = [l.rstrip() for l in fd]

    script_name = os.path.basename(__file__)
    first_line = input_lines[0] if input_lines else ""
    if 'autogenerated' in first_line and script_name not in first_line:
        common.warn("Skipping test which wasn't autogenerated by " +
                    script_name + ": " + test)
        return

    if args.update_only:
      if not first_line or 'autogenerated' not in first_line:
        common.warn("Skipping test which isn't autogenerated: " + test)
        return

    triple_in_ir = find_triple_in_ir(input_lines, args.verbose)
    run_lines = common.find_run_lines(test, input_lines)
    run_list, common_prefixes = build_run_list(test, run_lines, args.verbose)

    simple_functions = find_functions_with_one_bb(input_lines, args.verbose)

    func_dict = {}
    for run in run_list:
        for prefix in run.prefixes:
            func_dict.update({prefix: dict()})
    for prefixes, llc_args, triple_in_cmd in run_list:
        log('Extracted LLC cmd: llc {}'.format(llc_args), args.verbose)
        log('Extracted FileCheck prefixes: {}'.format(prefixes), args.verbose)

        raw_tool_output = args.llc(llc_args, test)
        if not triple_in_cmd and not triple_in_ir:
            common.warn('No triple found: skipping file', test_file=test)
            return

        build_function_body_dictionary(test, raw_tool_output,
                                       triple_in_cmd or triple_in_ir,
                                       prefixes, func_dict, args.verbose)

    state = 'toplevel'
    func_name = None
    prefix_set = set([prefix for run in run_list for prefix in run.prefixes])
    log('Rewriting FileCheck prefixes: {}'.format(prefix_set), args.verbose)

    if args.remove_common_prefixes:
        prefix_set.update(common_prefixes)
    elif common_prefixes:
        common.warn('Ignoring common prefixes: {}'.format(common_prefixes),
             test_file=test)

    comment_char = '#' if test.endswith('.mir') else ';'
    autogenerated_note = ('{} NOTE: Assertions have been autogenerated by '
                          'utils/{}'.format(comment_char, script_name))
    output_lines = []
    output_lines.append(autogenerated_note)

    for input_line in input_lines:
        if input_line == autogenerated_note:
            continue

        if state == 'toplevel':
            m = IR_FUNC_NAME_RE.match(input_line)
            if m:
                state = 'ir function prefix'
                func_name = m.group('func')
            if input_line.rstrip('| \r\n') == '---':
                state = 'document'
            output_lines.append(input_line)
        elif state == 'document':
            m = MIR_FUNC_NAME_RE.match(input_line)
            if m:
                state = 'mir function metadata'
                func_name = m.group('func')
            if input_line.strip() == '...':
                state = 'toplevel'
                func_name = None
            if should_add_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == 'mir function metadata':
            if should_add_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
            m = MIR_BODY_BEGIN_RE.match(input_line)
            if m:
                if func_name in simple_functions:
                    # If there's only one block, put the checks inside it
                    state = 'mir function prefix'
                    continue
                state = 'mir function body'
                add_checks_for_function(test, output_lines, run_list,
                                        func_dict, func_name, single_bb=False,
                                        verbose=args.verbose)
        elif state == 'mir function prefix':
            m = MIR_PREFIX_DATA_RE.match(input_line)
            if not m:
                state = 'mir function body'
                add_checks_for_function(test, output_lines, run_list,
                                        func_dict, func_name, single_bb=True,
                                        verbose=args.verbose)

            if should_add_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == 'mir function body':
            if input_line.strip() == '...':
                state = 'toplevel'
                func_name = None
            if should_add_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == 'ir function prefix':
            m = IR_PREFIX_DATA_RE.match(input_line)
            if not m:
                state = 'ir function body'
                add_checks_for_function(test, output_lines, run_list,
                                        func_dict, func_name, single_bb=False,
                                        verbose=args.verbose)

            if should_add_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == 'ir function body':
            if input_line.strip() == '}':
                state = 'toplevel'
                func_name = None
            if should_add_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)


    log('Writing {} lines to {}...'.format(len(output_lines), test), args.verbose)

    with open(test, 'wb') as fd:
        fd.writelines(['{}\n'.format(l).encode('utf-8') for l in output_lines])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--llc-binary', dest='llc', default='llc', type=LLC,
                        help='The "llc" binary to generate the test case with')
    parser.add_argument('--remove-common-prefixes', action='store_true',
                        help='Remove existing check lines whose prefixes are '
                             'shared between multiple commands')
    parser.add_argument('tests', nargs='+')
    args = common.parse_commandline_args(parser)

    test_paths = [test for pattern in args.tests for test in glob.glob(pattern)]
    for test in test_paths:
        try:
            update_test_file(args, test)
        except Exception:
            common.warn('Error processing file', test_file=test)
            raise


if __name__ == '__main__':
  main()
