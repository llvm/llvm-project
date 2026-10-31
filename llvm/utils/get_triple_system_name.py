#!/usr/bin/env python3
"""Derive a CMake system name from a target triple.

The recognized OS and environment names are defined in
llvm/include/llvm/TargetParser/TripleName.def. This script parses that
file so the CMake runtimes build can map a (possibly unnormalized)
triple to a CMake system name without running a compiled tool. CMake
invokes it once per target with --triple and reads the printed name.

Modes:
  --triple TRIPLE   Print the CMake system name for TRIPLE (empty if unknown or
                    the triple has too few components; the caller then falls back
                    to the host system name).

Options:
  --def-file PATH   Path to TripleName.def to parse instead of the copy located
                    relative to this script (in llvm/include/llvm/TargetParser).

"""

import argparse
import os
import re
import sys

_OS_RE = re.compile(r'^\s*TRIPLE_OS\(\s*(\w+)\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)')
_OS_ALIAS_RE = re.compile(r'^\s*TRIPLE_OS_ALIAS\(\s*(\w+)\s*,\s*"([^"]*)"\s*\)')
_ENV_RE = re.compile(r'^\s*TRIPLE_ENV\(\s*(\w+)\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)')


class TripleNames:
    def __init__(self):
        # Ordered lists of (prefix, value) preserving .def order, which matters
        # for prefix matching (longer prefixes precede shorter ones).
        self.os = []  # (name, cmake_name)
        self.env = []  # (name, cmake_override)


def find_def_file():
    # This script lives in llvm/utils/, the .def in llvm/include/...
    llvm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(llvm_dir, "include", "llvm", "TargetParser", "TripleName.def")


def parse(def_path):
    names = TripleNames()
    with open(def_path) as f:
        lines = f.readlines()

    # First pass: map each OS enum to its CMake name so aliases can reuse it.
    os_cmake = {}
    for line in lines:
        if m := _OS_RE.match(line):
            enum, _name, cmake_name = m.groups()
            os_cmake[enum] = cmake_name

    # Second pass: build the ordered lists (order matters for prefix matching).
    for line in lines:
        if m := _OS_RE.match(line):
            _enum, name, cmake_name = m.groups()
            names.os.append((name, cmake_name))
        elif m := _OS_ALIAS_RE.match(line):
            enum, alias = m.groups()
            names.os.append((alias, os_cmake.get(enum, "Generic")))
        elif m := _ENV_RE.match(line):
            _enum, name, override = m.groups()
            names.env.append((name, override))
    return names


def system_name(names, triple):
    # Mirror the component classification of Triple::normalize: do not
    # trust fixed positions. A triple may omit the vendor
    # (aarch64-linux-android), put the os in the vendor slot
    # (wasm32-wasi, i686-linux), or even be a bare os with no arch at
    # all (clang accepts --target=darwin as unknown-unknown-darwin),
    # so scan every component and use the first that classifies. No
    # arch name is a prefix of any os/env name, so scanning the first
    # component too cannot misclassify a real arch. This makes the
    # result match what normalize would produce even for unnormalized
    # inputs.
    parts = triple.split("-")

    # Environment override takes precedence over the OS mapping
    # (e.g. android wins over the linux in aarch64-linux-android), so
    # scan for it first.
    for comp in parts:
        for name, override in names.env:
            if override and comp.startswith(name):
                return override

    # Find the first component that classifies as an OS. A specific
    # (non-Generic) name wins immediately. A "Generic" match means the
    # component is a recognized OS with no CMake equivalent.
    found_generic_os = False
    for comp in parts:
        matched = False
        for name, cmake_name in names.os:
            if comp.startswith(name):
                if cmake_name != "Generic":
                    return cmake_name
                found_generic_os = True
                matched = True
                break
        if matched:
            break

    # Handle special case legacy os spellings that normalize rewrites
    # to windows.
    for comp in parts:
        if comp.startswith("mingw"):
            return "Windows"
        if comp.startswith("cygwin") or comp.startswith("msys"):
            return "CYGWIN"

    # Apple vendor triples with no specific CMake OS name (e.g.
    # firmware) map to Darwin.
    if len(parts) > 1 and parts[1] == "apple":
        return "Darwin"

    if found_generic_os:
        return "Generic"

    # No OS could be identified. Too few components to have one -> let
    # the caller fall back to the host system name.
    if len(parts) < 3:
        return None
    return "Generic"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--def-file", default=None, help="Path to TripleName.def")
    ap.add_argument(
        "--triple", required=True, help="Print the CMake system name for this triple"
    )
    args = ap.parse_args()

    def_path = args.def_file or find_def_file()
    names = parse(def_path)

    result = system_name(names, args.triple)
    print(result if result is not None else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
