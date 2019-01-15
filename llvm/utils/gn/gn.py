#!/usr/bin/env python
"""Calls `gn` with the right --dotfile= and --root= arguments for LLVM."""

# GN normally expects a file called '.gn' at the root of the repository.
# Since LLVM's GN build isn't supported, putting that file at the root
# is deemed inappropriate, which requires passing --dotfile= and -root= to GN.
# Since that gets old fast, this script automatically passes these arguments.

import os
import subprocess
import sys


THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, '..', '..', '..')


def main():
    # Find real gn executable. For now, just assume it's on PATH.
    # FIXME: Probably need to append '.exe' on Windows.
    gn = 'gn'

    # Compute --dotfile= and --root= args to add.
    extra_args = []
    gn_main_arg = next((x for x in sys.argv[1:] if not x.startswith('-')), None)
    if gn_main_arg != 'help':  # `gn help` gets confused by the switches.
        cwd = os.getcwd()
        dotfile = os.path.relpath(os.path.join(THIS_DIR, '.gn'), cwd)
        root = os.path.relpath(ROOT_DIR, cwd)
        extra_args = [ '--dotfile=' + dotfile, '--root=' + root ]

    # Run GN command with --dotfile= and --root= added.
    cmd = [gn] + extra_args + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
