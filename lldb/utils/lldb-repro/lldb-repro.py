#!/usr/bin/env python
"""lldb-repro

lldb-repro is a utility to transparently capture and replay debugger sessions
through the command line driver. Its used to test the reproducers by running
the test suite twice.

During the first run, with 'capture' as its first argument, it captures a
reproducer for every lldb invocation and saves it to a well-know location
derived from the arguments and current working directory.

During the second run, with 'replay' as its first argument, the test suite is
run again but this time every invocation of lldb replays the previously
recorded session.
"""

import sys
import os
import tempfile
import subprocess


def help():
    print("usage: {} capture|replay [args]".fmt(sys.argv[0]))


def main():
    if len(sys.argv) < 3:
        help()
        return 1

    # Compute a hash based on the input arguments and the current working
    # directory.
    args = ' '.join(sys.argv[3:])
    cwd = os.getcwd()
    input_hash = str(hash((cwd, args)))

    # Use the hash to "uniquely" identify a reproducer path.
    reproducer_path = os.path.join(tempfile.gettempdir(), input_hash)

    # Create a new lldb invocation with capture or replay enabled.
    lldb = os.path.join(os.path.dirname(sys.argv[0]), 'lldb')
    new_args = [sys.argv[1]]
    if sys.argv[2] == "replay":
        new_args.extend(['--replay', reproducer_path])
    elif sys.argv[2] == "capture":
        new_args.extend([
            '--capture', '--capture-path', reproducer_path,
            '--reproducer-auto-generate'
        ])
        new_args.extend(sys.argv[1:])
    else:
        help()
        return 1

    return subprocess.call(new_args)


if __name__ == '__main__':
    exit(main())
