#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A test case update script.

This script is a utility to update LLVM 'llubi' based test cases with new
FileCheck patterns.
"""

from __future__ import print_function

from sys import stderr
from traceback import print_exc
import argparse
import os
import subprocess
import sys

from UpdateTestChecks import common


# Invoke the tool that is being tested.
def invoke_tool(exe, cmd_args, ir, check_rc):
    with open(ir) as ir_file:
        substitutions = common.getSubstitutions(ir)
        stdout = subprocess.run(
            exe + " " + common.applySubstitutions(cmd_args, substitutions),
            shell=True,
            stdin=ir_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=check_rc,
        ).stdout.decode()
    # Fix line endings to unix CR style.
    return stdout.replace("\r\n", "\n")


def update_test(ti: common.TestInfo):
    if len(ti.run_lines) == 0:
        common.warn("No RUN lines found in test: " + ti.path)
        return
    if len(ti.run_lines) > 1:
        common.warn("Multiple RUN lines found in test: " + ti.path)
        common.warn("Only the first RUN line will be processed.")

    l = ti.run_lines[0]
    if "|" not in l:
        common.warn("Skipping unparsable RUN line: " + l)
        return

    commands = [cmd.strip() for cmd in l.split("|")]
    assert len(commands) == 2
    llubi_cmd = commands[-2]
    filecheck_cmd = commands[-1]
    args = llubi_cmd.split(" ")
    llubi_tool = args[0]
    check_rc = True
    if len(args) > 1 and args[0] == "not":
        llubi_tool = args[1]
        check_rc = False

    common.verify_filecheck_prefixes(filecheck_cmd)

    if llubi_tool != "llubi":
        common.warn("Skipping non-llubi RUN line: " + l)
        return

    if not filecheck_cmd.startswith("FileCheck "):
        common.warn("Skipping non-FileChecked RUN line: " + l)
        return

    llubi_args = llubi_cmd[llubi_cmd.index(llubi_tool) + len(llubi_tool) :].strip()
    llubi_args = llubi_args.replace("< %s", "").replace("%s", "").strip()
    prefixes = common.get_check_prefixes(filecheck_cmd)

    common.debug("Extracted llubi cmd:", llubi_tool, llubi_args)
    common.debug("Extracted FileCheck prefixes:", str(prefixes))
    prefix_set = set([prefix for prefix in prefixes])

    raw_tool_output = invoke_tool(
        ti.args.llubi_binary or llubi_tool,
        llubi_args,
        ti.path,
        check_rc=check_rc,
    )
    if ti.args.llubi_binary:
        raw_tool_output = raw_tool_output.replace(ti.args.llubi_binary, llubi_tool)

    output_lines = []
    common.dump_input_lines(output_lines, ti, prefix_set, ";")
    tool_output_lines = raw_tool_output.splitlines()
    if len(tool_output_lines) == 0:
        common.warn("No output from llubi.")
    else:
        output_lines.append("; CHECK: " + tool_output_lines[0])
        output_lines.extend(["; CHECK-NEXT: " + line for line in tool_output_lines[1:]])

    common.debug("Writing %d lines to %s..." % (len(output_lines), ti.path))
    with open(ti.path, "wb") as f:
        f.writelines(["{}\n".format(l).encode("utf-8") for l in output_lines])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llubi-binary",
        default=None,
        help='The "llubi" binary to use to generate the test case',
    )
    parser.add_argument(
        "--tool",
        default=None,
        help="Treat the given tool name as an llubi-like tool for which check lines should be generated",
    )
    parser.add_argument("tests", nargs="+")
    initial_args = common.parse_commandline_args(parser)

    script_name = os.path.basename(__file__)

    returncode = 0
    for ti in common.itertests(
        initial_args.tests, parser, script_name="utils/" + script_name
    ):
        try:
            update_test(ti)
        except Exception as e:
            stderr.write(f"Error: Failed to update test {ti.path}\n")
            print_exc()
            returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main())
