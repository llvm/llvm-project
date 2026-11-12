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
from libomputils import error, ScriptError, print_error_line


class DllExports(object):
    def __init__(self):
        self.filename = None
        self.exports = {}
        self.ordinals = set([])

    def add_uppercase_entries(self):
        # Ignored entries are C/C++ only functions
        ignores = [
            "omp_alloc",
            "omp_free",
            "omp_calloc",
            "omp_realloc",
            "omp_aligned_alloc",
            "omp_aligned_calloc",
        ]
        keys = list(self.exports.keys())
        for entry in keys:
            info = self.exports[entry]
            if info["obsolete"] or info["is_data"] or entry in ignores:
                continue
            if entry.startswith("omp_") or entry.startswith("kmp_"):
                newentry = entry.upper()
                if info["ordinal"]:
                    newordinal = info["ordinal"] + 1000
                else:
                    newordinal = None
                self.exports[newentry] = {
                    "obsolete": False,
                    "is_data": False,
                    "ordinal": newordinal,
                }

    @staticmethod
    def create(inputFile, defs=None):
        """Creates DllExports object from inputFile"""
        dllexports = DllExports()
        dllexports.filename = inputFile
        # Create a (possibly empty) list of definitions
        if defs:
            definitions = set(list(defs))
        else:
            definitions = set([])
        # Different kinds of lines to parse
        kw = r"[a-zA-Z_][a-zA-Z0-9_]*"
        ifndef = re.compile(r"%ifndef\s+({})".format(kw))
        ifdef = re.compile(r"%ifdef\s+({})".format(kw))
        endif = re.compile(r"%endif")
        export = re.compile(r"(-)?\s*({0})(=({0}))?(\s+([0-9]+|DATA))?".format(kw))

        def err(fil, num, msg):
            error("{}: {}: {}".format(fil, num, msg))

        defs_stack = []
        with open(inputFile) as f:
            for lineNumber, line in enumerate(f):
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Skip comment lines
                if line.startswith("#"):
                    continue
                # Encountered %ifndef DEF
                m = ifndef.search(line)
                if m:
                    defs_stack.append(m.group(1) not in definitions)
                    continue
                # Encountered %ifdef DEF
                m = ifdef.search(line)
                if m:
                    defs_stack.append(m.group(1) in definitions)
                    continue
                # Encountered %endif
                m = endif.search(line)
                if m:
                    if not defs_stack:
                        err(inputFile, lineNumber, "orphan %endif directive")
                    defs_stack.pop()
                    continue
                # Skip lines when not all %ifdef or %ifndef are true
                if defs_stack and not all(defs_stack):
                    continue
                # Encountered an export line
                m = export.search(line)
                if m:
                    obsolete = m.group(1) is not None
                    entry = m.group(2)
                    rename = m.group(4)
                    ordinal = m.group(6)
                    if entry in dllexports.exports:
                        err(
                            inputFile,
                            lineNumber,
                            "already specified entry: {}".format(entry),
                        )
                    if rename:
                        entry += "={}".format(rename)
                    # No ordinal number nor DATA specified
                    if not ordinal:
                        ordinal = None
                        is_data = False
                    # DATA ordinal
                    elif ordinal == "DATA":
                        ordinal = None
                        is_data = True
                    # Regular ordinal number
                    else:
                        is_data = False
                        try:
                            ordinal = int(ordinal)
                        except:
                            err(
                                inputFile,
                                lineNumber,
                                "Bad ordinal value: {}".format(ordinal),
                            )
                        if ordinal >= 1000 and (
                            entry.startswith("omp_") or entry.startswith("kmp_")
                        ):
                            err(
                                inputFile,
                                lineNumber,
                                "Ordinal of user-callable entry must be < 1000",
                            )
                        if ordinal >= 1000 and ordinal < 2000:
                            err(
                                inputFile,
                                lineNumber,
                                "Ordinals between 1000 and 1999 are reserved.",
                            )
                        if ordinal in dllexports.ordinals:
                            err(
                                inputFile,
                                lineNumber,
                                "Ordinal {} has already been used.".format(ordinal),
                            )
                    dllexports.exports[entry] = {
                        "ordinal": ordinal,
                        "obsolete": obsolete,
                        "is_data": is_data,
                    }
                    continue
                err(
                    inputFile,
                    lineNumber,
                    'Cannot parse line:{}"{}"'.format(os.linesep, line),
                )
        if defs_stack:
            error("syntax error: Unterminated %if directive")
        return dllexports


def generate_def(dllexports, f, no_ordinals=False, name=None):
    """Using dllexports data, write the exports to file, f"""
    if name:
        f.write("LIBRARY {}\n".format(name))
    f.write("EXPORTS\n")
    for entry in sorted(list(dllexports.exports.keys())):
        info = dllexports.exports[entry]
        if info["obsolete"]:
            continue
        f.write("    {:<40} ".format(entry))
        if info["is_data"]:
            f.write("DATA\n")
        elif no_ordinals or not info["ordinal"]:
            f.write("\n")
        else:
            f.write("@{}\n".format(info["ordinal"]))


def main():
    parser = argparse.ArgumentParser(
        description="Reads input file of dllexports, processes conditional"
        " directives, checks content for consistency, and generates"
        " output file suitable for linker"
    )
    parser.add_argument(
        "-D",
        metavar="DEF",
        action="append",
        dest="defs",
        help="Define a variable. Can specify" " this more than once.",
    )
    parser.add_argument(
        "--no-ordinals",
        action="store_true",
        help="Specify that no ordinal numbers should be generated",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        help="Specify library name for def file LIBRARY statement",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        dest="output",
        help="Specify output file name. If not specified," " output is sent to stdout",
    )
    parser.add_argument("dllexports", help="The input file describing dllexports")
    commandArgs = parser.parse_args()
    defs = set([])
    if commandArgs.defs:
        defs = set(commandArgs.defs)
    dllexports = DllExports.create(commandArgs.dllexports, defs)
    dllexports.add_uppercase_entries()
    try:
        output = open(commandArgs.output, "w") if commandArgs.output else sys.stdout
        generate_def(dllexports, output, commandArgs.no_ordinals, commandArgs.name)
    finally:
        if commandArgs.output:
            output.close()


if __name__ == "__main__":
    try:
        main()
    except ScriptError as e:
        print_error_line(str(e))
        sys.exit(1)

# end of file
