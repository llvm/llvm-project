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
import platform
import re
import sys
from libomputils import (
    ScriptError,
    error,
    execute_command,
    print_info_line,
    print_error_line,
)


def get_deps_readelf(filename):
    """Get list of dependencies from readelf"""
    deps = []
    # Force readelf call to be in English
    os.environ["LANG"] = "C"
    r = execute_command(["readelf", "-d", filename])
    if r.returncode != 0:
        error("readelf -d {} failed".format(filename))
    neededRegex = re.compile(r"\(NEEDED\)\s+Shared library: \[([a-zA-Z0-9_.-]+)\]")
    for line in r.stdout.split(os.linesep):
        match = neededRegex.search(line)
        if match:
            deps.append(match.group(1))
    return deps


def get_deps_otool(filename):
    """Get list of dependencies from otool"""
    deps = []
    r = execute_command(["otool", "-L", filename])
    if r.returncode != 0:
        error("otool -L {} failed".format(filename))
    libRegex = re.compile(r"([^ \t]+)\s+\(compatibility version ")
    thisLibRegex = re.compile(r"@rpath/{}".format(os.path.basename(filename)))
    for line in r.stdout.split(os.linesep):
        match = thisLibRegex.search(line)
        if match:
            # Don't include the library itself as a needed dependency
            continue
        match = libRegex.search(line)
        if match:
            deps.append(match.group(1))
            continue
    return deps


def get_deps_link(filename):
    """Get list of dependecies from link (Windows OS)"""
    depsSet = set([])
    f = filename.lower()
    args = ["link", "/DUMP"]
    if f.endswith(".lib"):
        args.append("/DIRECTIVES")
    elif f.endswith(".dll") or f.endswith(".exe"):
        args.append("/DEPENDENTS")
    else:
        error("unrecognized file extension: {}".format(filename))
    args.append(filename)
    r = execute_command(args)
    if r.returncode != 0:
        error("{} failed".format(args.command))
    if f.endswith(".lib"):
        regex = re.compile(r"\s*[-/]defaultlib:(.*)\s*$")
        for line in r.stdout.split(os.linesep):
            line = line.lower()
            match = regex.search(line)
            if match:
                depsSet.add(match.group(1))
    else:
        started = False
        markerStart = re.compile(r"Image has the following depend")
        markerEnd = re.compile(r"Summary")
        markerEnd2 = re.compile(r"Image has the following delay load depend")
        for line in r.stdout.split(os.linesep):
            if not started:
                if markerStart.search(line):
                    started = True
                    continue
            else:  # Started parsing the libs
                line = line.strip()
                if not line:
                    continue
                if markerEnd.search(line) or markerEnd2.search(line):
                    break
                depsSet.add(line.lower())
    return list(depsSet)


def main():
    parser = argparse.ArgumentParser(description="Check library dependencies")
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Produce plain, bare output: just a list"
        " of libraries, a library per line",
    )
    parser.add_argument(
        "--expected",
        metavar="CSV_LIST",
        help="CSV_LIST is a comma-separated list of expected"
        ' dependencies (or "none"). checks the specified'
        " library has only expected dependencies.",
    )

    parser.add_argument("library", help="The library file to check")
    commandArgs = parser.parse_args()
    # Get dependencies
    deps = []

    system = platform.system()
    if system == "Windows":
        deps = get_deps_link(commandArgs.library)
    elif system == "Darwin":
        deps = get_deps_otool(commandArgs.library)
    else:
        deps = get_deps_readelf(commandArgs.library)
    deps = sorted(deps)

    # If bare output specified, then just print the dependencies one per line
    if commandArgs.bare:
        print(os.linesep.join(deps))
        return

    # Calculate unexpected dependencies if expected list specified
    unexpected = []
    if commandArgs.expected:
        # none => any dependency is unexpected
        if commandArgs.expected == "none":
            unexpected = list(deps)
        else:
            expected = [d.strip() for d in commandArgs.expected.split(",")]
            unexpected = [d for d in deps if d not in expected]

    # Regular output
    print_info_line("Dependencies:")
    for dep in deps:
        print_info_line("    {}".format(dep))
    if unexpected:
        print_error_line("Unexpected Dependencies:")
        for dep in unexpected:
            print_error_line("    {}".format(dep))
        error("found unexpected dependencies")


if __name__ == "__main__":
    try:
        main()
    except ScriptError as e:
        print_error_line(str(e))
        sys.exit(1)

# end of file
