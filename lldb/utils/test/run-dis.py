#!/usr/bin/env python

"""
Run lldb disassembler on all the binaries specified by a combination of root dir
and path pattern.
"""

from __future__ import print_function

import os
import sys
import subprocess
import re
from optparse import OptionParser

# The directory of this Python script as well as the lldb-disasm.py workhorse.
scriptPath = None

# The root directory for the SDK symbols.
root_dir = None

# The regular expression pattern to match the desired pathname to the binaries.
path_pattern = None

# And the re-compiled regular expression object.
path_regexp = None

# If specified, number of symbols to disassemble for each qualified binary.
num_symbols = -1

# Command template of the invocation of lldb disassembler.
template = '%s/lldb-disasm.py -C "platform select remote-ios" -o "-n" -q -e %s -n %s'

# Regular expression for detecting file output for Mach-o binary.
mach_o = re.compile('\sMach-O.+binary')


def isbinary(path):
    file_output = subprocess.Popen(["file", path],
                                   stdout=subprocess.PIPE).stdout.read()
    return (mach_o.search(file_output) is not None)


def walk_and_invoke(sdk_root, path_regexp, suffix, num_symbols):
    """Look for matched file and invoke lldb disassembly on it."""
    global scriptPath

    for root, dirs, files in os.walk(sdk_root, topdown=False):
        for name in files:
            path = os.path.join(root, name)

            # We're not interested in .h file.
            if name.endswith(".h"):
                continue
            # Neither a symbolically linked file.
            if os.path.islink(path):
                continue

            # We'll be pattern matching based on the path relative to the SDK
            # root.
            replaced_path = path.replace(root_dir, "", 1)
            # Check regular expression match for the replaced path.
            if not path_regexp.search(replaced_path):
                continue
            # If a suffix is specified, check it, too.
            if suffix and not name.endswith(suffix):
                continue
            if not isbinary(path):
                continue

            command = template % (
                scriptPath, path, num_symbols if num_symbols > 0 else 1000)
            print("Running %s" % (command))
            os.system(command)


def main():
    """Read the root dir and the path spec, invoke lldb-disasm.py on the file."""
    global scriptPath
    global root_dir
    global path_pattern
    global path_regexp
    global num_symbols

    scriptPath = sys.path[0]

    parser = OptionParser(usage="""\
Run lldb disassembler on all the binaries specified by a combination of root dir
and path pattern.
""")
    parser.add_option(
        '-r',
        '--root-dir',
        type='string',
        action='store',
        dest='root_dir',
        help='Mandatory: the root directory for the SDK symbols.')
    parser.add_option(
        '-p',
        '--path-pattern',
        type='string',
        action='store',
        dest='path_pattern',
        help='Mandatory: regular expression pattern for the desired binaries.')
    parser.add_option('-s', '--suffix',
                      type='string', action='store', default=None,
                      dest='suffix',
                      help='Specify the suffix of the binaries to look for.')
    parser.add_option(
        '-n',
        '--num-symbols',
        type='int',
        action='store',
        default=-1,
        dest='num_symbols',
        help="""The number of symbols to disassemble, if specified.""")

    opts, args = parser.parse_args()
    if not opts.root_dir or not opts.path_pattern:
        parser.print_help()
        sys.exit(1)

    # Sanity check the root directory.
    root_dir = opts.root_dir
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        parser.print_help()
        sys.exit(1)

    path_pattern = opts.path_pattern
    path_regexp = re.compile(path_pattern)
    suffix = opts.suffix
    num_symbols = opts.num_symbols

    print("Root directory for SDK symbols:", root_dir)
    print("Regular expression for the binaries:", path_pattern)
    print("Suffix of the binaries to look for:", suffix)
    print("num of symbols to disassemble:", num_symbols)

    walk_and_invoke(root_dir, path_regexp, suffix, num_symbols)


if __name__ == '__main__':
    main()
