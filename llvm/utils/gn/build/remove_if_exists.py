#!/usr/bin/env python3

"""Removes a given file, if it exists."""

import argparse
import errno
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stamp", required=True, help="name of a file whose mtime is updated on run"
    )
    parser.add_argument("file")
    args = parser.parse_args()

    try:
        os.remove(args.file)
    except FileNotFoundError:
        pass

    open(args.stamp, "w")  # Update mtime on stamp file.


if __name__ == "__main__":
    sys.exit(main())
