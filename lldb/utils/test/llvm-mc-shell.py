#!/usr/bin/env python

"""
Run llvm-mc interactively.

"""

import os
import sys
from optparse import OptionParser


def is_exe(fpath):
    """Check whether fpath is an executable."""
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    """Find the full path to a program, or return None."""
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def llvm_mc_loop(mc, mc_options):
    contents = []
    fname = 'mc-input.txt'
    sys.stdout.write(
        "Enter your input to llvm-mc.  A line starting with 'END' terminates the current batch of input.\n")
    sys.stdout.write("Enter 'quit' or Ctrl-D to quit the program.\n")
    while True:
        sys.stdout.write("> ")
        next = sys.stdin.readline()
        # EOF => terminate this llvm-mc shell
        if not next or next.startswith('quit'):
            sys.stdout.write('\n')
            sys.exit(0)
        # 'END' => send the current batch of input to llvm-mc
        if next.startswith('END'):
            # Write contents to our file and clear the contents.
            with open(fname, 'w') as f:
                f.writelines(contents)
                # Clear the list: replace all items with an empty list.
                contents[:] = []

            # Invoke llvm-mc with our newly created file.
            mc_cmd = '%s %s %s' % (mc, mc_options, fname)
            sys.stdout.write("Executing command: %s\n" % mc_cmd)
            os.system(mc_cmd)
        else:
            # Keep accumulating our input.
            contents.append(next)


def main():
    # This is to set up the Python path to include the pexpect-2.4 dir.
    # Remember to update this when/if things change.
    scriptPath = sys.path[0]
    sys.path.append(
        os.path.join(
            scriptPath,
            os.pardir,
            os.pardir,
            'test',
            'pexpect-2.4'))

    parser = OptionParser(usage="""\
Do llvm-mc interactively within a shell-like environment.  A batch of input is
submitted to llvm-mc to execute whenever you terminate the current batch by
inputing a line which starts with 'END'.  Quit the program by either 'quit' or
Ctrl-D.

Usage: %prog [options]
""")
    parser.add_option('-m', '--llvm-mc',
                      type='string', action='store',
                      dest='llvm_mc',
                      help="""The llvm-mc executable full path, if specified.
                      Otherwise, it must be present in your PATH environment.""")

    parser.add_option(
        '-o',
        '--options',
        type='string',
        action='store',
        dest='llvm_mc_options',
        help="""The options passed to 'llvm-mc' command if specified.""")

    opts, args = parser.parse_args()

    llvm_mc = opts.llvm_mc if opts.llvm_mc else which('llvm-mc')
    if not llvm_mc:
        parser.print_help()
        sys.exit(1)

    # This is optional.  For example:
    # --options='-disassemble -triple=arm-apple-darwin -debug-only=arm-disassembler'
    llvm_mc_options = opts.llvm_mc_options

    # We have parsed the options.
    print "llvm-mc:", llvm_mc
    print "llvm-mc options:", llvm_mc_options

    llvm_mc_loop(llvm_mc, llvm_mc_options)

if __name__ == '__main__':
    main()
