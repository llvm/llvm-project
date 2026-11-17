#!/usr/bin/env python3

"""Syncs files from a source dir to an output dir.

Reads a list of files from a response file, copies them from the source dir
to the output dir, and removes files in the output dir that are not in the
list (except for files passed via --except)."""

import argparse
import os
import shlex
import sys


def read(filename):
    with open(filename) as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stamp", required=True, help="name of a file whose mtime is updated on run"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--except", dest="exceptions", action="append", default=[])
    parser.add_argument("rspfile")
    args = parser.parse_args()

    files = shlex.split(read(args.rspfile))

    # Copy files from source dir to output dir.
    for f in files:
        src = os.path.join(args.source_dir, f)
        dst = os.path.join(args.output_dir, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = read(src)
        if not os.path.exists(dst) or read(dst) != data:
            with open(dst, "w") as dst_file:
                dst_file.write(data)

    # Remove files in output dir that are not in the list.
    want = set(files)
    exceptions = set(args.exceptions)
    for root, dirs, filenames in os.walk(args.output_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, args.output_dir)
            if relpath not in want and relpath not in exceptions:
                os.remove(filepath)

    open(args.stamp, "w")  # Update mtime on stamp file.


if __name__ == "__main__":
    sys.exit(main())
