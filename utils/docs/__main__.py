# -*- coding: utf-8 -*-

"""Command-line entrypoint to utils/docs

Use this as e.g. `python utils/docs --test` to run docs smoke tests.
"""

import sys
import argparse
import subprocess
from pathlib import Path

from llvm_sphinx.ext import absolute_links, ghlinks
from typing import List


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run sphinx self-tests")
    parser.add_argument(
        "--test-component",
        choices=("absolute-links", "ghlinks"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.test:
        script = Path(__file__).resolve()
        for component in ("absolute-links", "ghlinks"):
            subprocess.run(
                [sys.executable, str(script), "--test-component", component],
                check=True,
            )
        print("llvm_sphinx: tests passed; next, rebuild the affected project docs")
        return 0

    if args.test_component == "absolute-links":
        absolute_links.run_tests()
        return 0
    if args.test_component == "ghlinks":
        ghlinks.run_tests()
        return 0

    parser.print_help(sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
