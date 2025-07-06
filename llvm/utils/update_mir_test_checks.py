#!/usr/bin/env python3

"""Updates FileCheck checks in MIR tests.

This script is a utility to update MIR based tests with new FileCheck
patterns.

The checks added by this script will cover the entire body of each
function it handles. Virtual registers used are given names via
FileCheck patterns, so if you do want to check a subset of the body it
should be straightforward to trim out the irrelevant parts. None of
the YAML metadata will be checked, other than function names, and fixedStack
if the --print-fixed-stack option is used.

If there are multiple llc commands in a test, the full set of checks
will be repeated for each different check pattern. Checks for patterns
that are common between different commands will be left as-is by
default, or removed if the --remove-common-prefixes flag is provided.
"""

from __future__ import print_function

from sys import stderr
from traceback import print_exc
import argparse
import collections
import glob
import os
import re
import subprocess
import sys

from UpdateTestChecks import common

VREG_RE = re.compile(r"(%[0-9]+)(?:\.[a-z0-9_]+)?(?::[a-z0-9_]+)?(?:\([<>a-z0-9 ]+\))?")
MI_FLAGS_STR = (
    r"(frame-setup |frame-destroy |nnan |ninf |nsz |arcp |contract |afn "
    r"|reassoc |nuw |nsw |exact |nofpexcept |nomerge |disjoint )*"
)
VREG_DEF_FLAGS_STR = r"(?:dead |undef )*"
VREG_DEF_RE = re.compile(
    r"^ *(?P<vregs>{2}{0}(?:, {2}{0})*) = "
    r"{1}(?P<opcode>[A-Zt][A-Za-z0-9_]+)".format(
        VREG_RE.pattern, MI_FLAGS_STR, VREG_DEF_FLAGS_STR
    )
)

MIR_FUNC_RE = re.compile(
    r"^---$"
    r"\n"
    r"^ *name: *(?P<func>[A-Za-z0-9_.-]+)$"
    r".*?"
    r"(?:^ *fixedStack: *(\[\])? *\n"
    r"(?P<fixedStack>.*?)\n?"
    r"^ *stack:"
    r".*?)?"
    r"^ *body: *\|\n"
    r"(?P<body>.*?)\n"
    r"^\.\.\.$",
    flags=(re.M | re.S),
)


class LLC:
    def __init__(self, bin):
        self.bin = bin

    def __call__(self, args, ir):
        if ir.endswith(".mir"):
            args = "{} -x mir".format(args)
        with open(ir) as ir_file:
            stdout = subprocess.check_output(
                "{} {}".format(self.bin, args), shell=True, stdin=ir_file
            )
            if sys.version_info[0] > 2:
                stdout = stdout.decode()
            # Fix line endings to unix CR style.
            stdout = stdout.replace("\r\n", "\n")
        return stdout


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
        if "|" not in l:
            common.warn("Skipping unparsable RUN line: " + l)
            continue

        commands = [cmd.strip() for cmd in l.split("|", 1)]
        llc_cmd = commands[0]
        filecheck_cmd = commands[1] if len(commands) > 1 else ""
        common.verify_filecheck_prefixes(filecheck_cmd)

        if not llc_cmd.startswith("llc "):
            common.warn("Skipping non-llc RUN line: {}".format(l), test_file=test)
            continue
        if not filecheck_cmd.startswith("FileCheck "):
            common.warn(
                "Skipping non-FileChecked RUN line: {}".format(l), test_file=test
            )
            continue

        triple = None
        m = common.TRIPLE_ARG_RE.search(llc_cmd)
        if m:
            triple = m.group(1)
        # If we find -march but not -mtriple, use that.
        m = common.MARCH_ARG_RE.search(llc_cmd)
        if m and not triple:
            triple = "{}--".format(m.group(1))

        cmd_args = llc_cmd[len("llc") :].strip()
        cmd_args = cmd_args.replace("< %s", "").replace("%s", "").strip()
        check_prefixes = common.get_check_prefixes(filecheck_cmd)
        all_prefixes += check_prefixes

        run_list.append((check_prefixes, cmd_args, triple))

    # Sort prefixes that are shared between run lines before unshared prefixes.
    # This causes us to prefer printing shared prefixes.
    for run in run_list:
        run[0].sort(key=lambda prefix: -all_prefixes.count(prefix))

    return run_list


def build_function_info_dictionary(
    test, raw_tool_output, triple, prefixes, func_dict, verbose
):
    for m in MIR_FUNC_RE.finditer(raw_tool_output):
        func = m.group("func")
        fixedStack = m.group("fixedStack")
        body = m.group("body")
        if verbose:
            log("Processing function: {}".format(func))
            for l in body.splitlines():
                log("  {}".format(l))

        # Vreg mangling
        mangled = []
        vreg_map = {}
        for func_line in body.splitlines(keepends=True):
            m = VREG_DEF_RE.match(func_line)
            if m:
                for vreg in VREG_RE.finditer(m.group("vregs")):
                    if vreg.group(1) in vreg_map:
                        name = vreg_map[vreg.group(1)]
                    else:
                        name = mangle_vreg(m.group("opcode"), vreg_map.values())
                        vreg_map[vreg.group(1)] = name
                    func_line = func_line.replace(
                        vreg.group(1), "[[{}:%[0-9]+]]".format(name), 1
                    )
            for number, name in vreg_map.items():
                func_line = re.sub(
                    r"{}\b".format(number), "[[{}]]".format(name), func_line
                )
            mangled.append(func_line)
        body = "".join(mangled)

        for prefix in prefixes:
            info = common.function_body(
                body, fixedStack, None, None, None, None, ginfo=None
            )
            if func in func_dict[prefix]:
                if (
                    not func_dict[prefix][func]
                    or func_dict[prefix][func].scrub != info.scrub
                    or func_dict[prefix][func].extrascrub != info.extrascrub
                ):
                    func_dict[prefix][func] = None
            else:
                func_dict[prefix][func] = info


def mangle_vreg(opcode, current_names):
    base = opcode
    # Simplify some common prefixes and suffixes
    if opcode.startswith("G_"):
        base = base[len("G_") :]
    if opcode.endswith("_PSEUDO"):
        base = base[: len("_PSEUDO")]
    # Shorten some common opcodes with long-ish names
    base = dict(
        IMPLICIT_DEF="DEF",
        GLOBAL_VALUE="GV",
        CONSTANT="C",
        FCONSTANT="C",
        MERGE_VALUES="MV",
        UNMERGE_VALUES="UV",
        INTRINSIC="INT",
        INTRINSIC_W_SIDE_EFFECTS="INT",
        INSERT_VECTOR_ELT="IVEC",
        EXTRACT_VECTOR_ELT="EVEC",
        SHUFFLE_VECTOR="SHUF",
    ).get(base, base)
    # Avoid ambiguity when opcodes end in numbers
    if len(base.rstrip("0123456789")) < len(base):
        base += "_"

    i = 0
    for name in current_names:
        if name.rstrip("0123456789") == base:
            i += 1
    if i:
        return "{}{}".format(base, i)
    return base


def update_test_file(args, test, autogenerated_note):
    with open(test) as fd:
        input_lines = [l.rstrip() for l in fd]

    triple_in_ir = find_triple_in_ir(input_lines, args.verbose)
    run_lines = common.find_run_lines(test, input_lines)
    run_list = build_run_list(test, run_lines, args.verbose)

    func_dict = {}
    for run in run_list:
        for prefix in run[0]:
            func_dict.update({prefix: dict()})
    for prefixes, llc_args, triple_in_cmd in run_list:
        log("Extracted LLC cmd: llc {}".format(llc_args), args.verbose)
        log("Extracted FileCheck prefixes: {}".format(prefixes), args.verbose)

        raw_tool_output = args.llc_binary(llc_args, test)
        if not triple_in_cmd and not triple_in_ir:
            common.warn("No triple found: skipping file", test_file=test)
            return

        build_function_info_dictionary(
            test,
            raw_tool_output,
            triple_in_cmd or triple_in_ir,
            prefixes,
            func_dict,
            args.verbose,
        )

    prefix_set = set([prefix for run in run_list for prefix in run[0]])
    log("Rewriting FileCheck prefixes: {}".format(prefix_set), args.verbose)

    output_lines = common.add_mir_checks(
        input_lines,
        prefix_set,
        autogenerated_note,
        test,
        run_list,
        func_dict,
        args.print_fixed_stack,
        first_check_is_next=False,
        at_the_function_name=False,
    )

    log("Writing {} lines to {}...".format(len(output_lines), test), args.verbose)

    with open(test, "wb") as fd:
        fd.writelines(["{}\n".format(l).encode("utf-8") for l in output_lines])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--llc-binary",
        default="llc",
        type=LLC,
        help='The "llc" binary to generate the test case with',
    )
    parser.add_argument(
        "--print-fixed-stack",
        action="store_true",
        help="Add check lines for fixedStack",
    )
    parser.add_argument("tests", nargs="+")
    args = common.parse_commandline_args(parser)

    script_name = os.path.basename(__file__)
    returncode = 0
    for ti in common.itertests(args.tests, parser, script_name="utils/" + script_name):
        try:
            update_test_file(ti.args, ti.path, ti.test_autogenerated_note)
        except Exception as e:
            stderr.write(f"Error: Failed to update test {ti.path}\n")
            print_exc()
            returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main())
