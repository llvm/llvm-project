#!/usr/bin/env python3
"""Generates .pyi stubs for nanobind extensions using nanobind's stubgen."""

import argparse
import ctypes
import importlib.util
import sys
from pathlib import Path

from python.runfiles import Runfiles


def load_extension(path: Path):
    """Load an extension module from a .so file with RTLD_GLOBAL."""
    module_name = path.stem.removesuffix(".abi3")

    # Load with RTLD_GLOBAL so symbols are available to dependent extensions.
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        sys.exit(f"Failed to load extension from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module", required=True, help="Module name to generate stubs for"
    )
    parser.add_argument(
        "--deps", required=True, help="Comma-separated .so files to load"
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    for dep_path in args.deps.split(","):
        load_extension(Path(dep_path).resolve())

    runfiles = Runfiles.Create()
    stubgen_path = runfiles.Rlocation("+llvm_repos_extension+nanobind/src/stubgen.py")
    spec = importlib.util.spec_from_file_location("stubgen", stubgen_path)
    stubgen = importlib.util.module_from_spec(spec)
    sys.modules["stubgen"] = stubgen
    spec.loader.exec_module(stubgen)
    stubgen.main(["-m", args.module, "-r", "-O", args.output])


if __name__ == "__main__":
    main()
