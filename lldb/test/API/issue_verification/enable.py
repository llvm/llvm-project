#!/usr/bin/env python
"""Renames *.py.park files to *.py."""
import os
import sys


def main():
    """Drives the main script behavior."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    for filename in os.listdir(script_dir):
        basename, extension = os.path.splitext(filename)
        if basename.startswith("Test") and extension == '.park':
            source_path = os.path.join(script_dir, filename)
            dest_path = os.path.join(script_dir, basename)
            sys.stdout.write("renaming {} to {}\n".format(
                source_path, dest_path))
            os.rename(source_path, dest_path)

if __name__ == "__main__":
    main()
