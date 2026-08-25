"""Expand @VAR@ placeholders in a template file.

The Bazel counterpart of CMake's configure_file(@ONLY) and GN's
write_cmake_config, so a template such as InstrumentorVariables.inc.in produces
identical output under all three build systems.

Values come from --substitution NAME=VALUE, or --substitution-file NAME=PATH to
take the value from a file's contents -- the analogue of GN's `@file:` syntax
and CMake's file(READ) + configure_file.
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--substitution", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument(
        "--substitution-file", action="append", default=[], metavar="NAME=PATH"
    )
    args = parser.parse_args(argv)

    values = {}
    for spec in args.substitution:
        name, _, value = spec.partition("=")
        values[name] = value
    for spec in args.substitution_file:
        name, _, path = spec.partition("=")
        values[name] = Path(path).read_text(encoding="utf-8")

    text = Path(args.template).read_text(encoding="utf-8")
    for name, value in values.items():
        text = text.replace("@%s@" % name, value)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
