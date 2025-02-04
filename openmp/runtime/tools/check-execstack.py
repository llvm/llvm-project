#!/usr/bin/env python3

#
# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for license information.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//
#

import argparse
import os
import re
import sys
from libomputils import ScriptError, error, print_error_line, execute_command


def is_stack_executable_readelf(library):
    """Returns true if stack of library file is executable"""
    r = execute_command(["readelf", "-l", "-W", library])
    if r.returncode != 0:
        error("{} failed".format(r.command))
    stack_lines = []
    for line in r.stdout.split(os.linesep):
        if re.search("STACK", line):
            stack_lines.append(line.strip())
    if not stack_lines:
        error("{}: Not stack segment found".format(library))
    if len(stack_lines) > 1:
        error("{}: More than one stack segment found".format(library))
    h = r"0x[0-9a-fA-F]+"
    m = re.search(
        r"((GNU_)?STACK)\s+({0})\s+({0})\s+({0})\s+({0})\s+({0})"
        " ([R ][W ][E ])".format(h),
        stack_lines[0],
    )
    if not m:
        error("{}: Cannot parse stack segment line".format(library))
    if m:
        flags = m.group(8)
        if "E" in flags:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check library does not have" " executable stack"
    )
    parser.add_argument("library", help="The library file to check")
    commandArgs = parser.parse_args()
    if is_stack_executable_readelf(commandArgs.library):
        error("{}: Stack is executable".format(commandArgs.library))


if __name__ == "__main__":
    try:
        main()
    except ScriptError as e:
        print_error_line(str(e))
        sys.exit(1)

# end of file
