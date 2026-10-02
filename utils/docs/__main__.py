# -*- coding: utf-8 -*-

"""Command-line entrypoint to utils/docs

Use this as e.g. `python utils/docs --test` to run all docs smoke tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from llvm_sphinx.ext import absolute_links, ghlinks


TEST_COMPONENTS = {
    "absolute-links": absolute_links.run_tests,
    "ghlinks": ghlinks.run_tests,
}


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        choices=TEST_COMPONENTS,
        metavar="COMPONENT",
        nargs="*",
        help="run selected Sphinx smoke tests (default: all components)",
    )
    args = parser.parse_args(argv)

    if args.test is None:
        parser.print_help(sys.stderr)
        return 0

    test_components = args.test or tuple(TEST_COMPONENTS)
    if len(test_components) > 1:
        script = Path(__file__).resolve()
        for component in test_components:
            subprocess.run(
                [sys.executable, str(script), "--test", component],
                check=True,
                stdout=subprocess.DEVNULL,
            )
    else:
        TEST_COMPONENTS[test_components[0]]()
    print("llvm_sphinx: tests passed; next, rebuild the affected project docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
