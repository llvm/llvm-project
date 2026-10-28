#!/usr/bin/env python3

import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: check_pdb_exists.py <directory> <name-file>",
            file=sys.stderr,
        )
        return 2

    directory = pathlib.Path(sys.argv[1])
    name_file = pathlib.Path(sys.argv[2])

    name = name_file.read_text().strip()
    path = directory / (name + ".pdb")
    assert path.is_file(), f"missing file: {path}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
