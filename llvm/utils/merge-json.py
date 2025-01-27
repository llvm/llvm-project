#!/usr/bin/env python
"""A command line utility to merge two JSON files.

This is a python program that merges two JSON files into a single one. The
intended use for this is to combine generated 'compile_commands.json' files
created by CMake when performing an LLVM runtime build.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        type=str,
        help="The output file to write JSON data to",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "json_files", type=str, nargs="+", help="Input JSON files to merge"
    )
    args = parser.parse_args()

    merged_data = []

    for json_file in args.json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                merged_data.extend(data)
        except (IOError, json.JSONDecodeError) as e:
            continue

    # LLVM passes this argument by default but it is not supported by clang,
    # causing annoying errors in the generated compile_commands.json file.
    # Remove it here before we deduplicate the entries.
    for entry in merged_data:
        if isinstance(entry, dict) and "command" in entry:
            entry["command"] = entry["command"].replace("-fno-lifetime-dse ", "")

    # Deduplicate by converting each entry to a tuple of sorted key-value pairs
    unique_data = list({json.dumps(entry, sort_keys=True) for entry in merged_data})
    unique_data = [json.loads(entry) for entry in unique_data]

    with open(args.o, "w") as f:
        json.dump(unique_data, f, indent=2)


if __name__ == "__main__":
    main()
