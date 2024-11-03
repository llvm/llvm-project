#!/usr/bin/env python3

# Generates a version script for an architecture so that it can be incorporated
# into gcc_s.ver.

from collections import defaultdict
from itertools import chain
import argparse, subprocess, sys, os


def split_suffix(symbol):
    """
    Splits a symbol such as `__gttf2@GCC_3.0` into a triple representing its
    function name (__gttf2), version name (GCC_3.0), and version number (300).

    The version number acts as a priority. Since earlier versions are more
    accessible and are likely to be used more, the lower the number is, the higher
    its priortiy. A symbol that has a '@@' instead of '@' has been designated by
    the linker as the default symbol, and is awarded a priority of -1.
    """
    if "@" not in symbol:
        return None
    data = [i for i in filter(lambda s: s, symbol.split("@"))]
    _, version = data[-1].split("_")
    version = version.replace(".", "")
    priority = -1 if "@@" in symbol else int(version + "0" * (3 - len(version)))
    return data[0], data[1], priority


def invert_mapping(symbol_map):
    """Transforms a map from Key->Value to Value->Key."""
    store = defaultdict(list)
    for symbol, (version, _) in symbol_map.items():
        store[version].append(symbol)
    result = []
    for k, v in store.items():
        v.sort()
        result.append((k, v))
    result.sort(key=lambda x: x[0])
    return result


def intersection(llvm, gcc):
    """
    Finds the intersection between the symbols extracted from compiler-rt.a/libunwind.a
    and libgcc_s.so.1.
    """
    common_symbols = {}
    for i in gcc:
        suffix_triple = split_suffix(i)
        if not suffix_triple:
            continue

        symbol, version_name, version_number = suffix_triple
        if symbol in llvm:
            if symbol not in common_symbols:
                common_symbols[symbol] = (version_name, version_number)
                continue
            if version_number < common_symbols[symbol][1]:
                common_symbols[symbol] = (version_name, version_number)
    return invert_mapping(common_symbols)


def find_function_names(path):
    """
    Runs readelf on a binary and reduces to only defined functions. Equivalent to
    `llvm-readelf --wide ${path} | grep 'FUNC' | grep -v 'UND' | awk '{print $8}'`.
    """
    result = subprocess.run(args=["llvm-readelf", "-su", path], capture_output=True)

    if result.returncode != 0:
        print(result.stderr.decode("utf-8"), file=sys.stderr)
        sys.exit(1)

    stdout = result.stdout.decode("utf-8")
    stdout = filter(lambda x: "FUNC" in x and "UND" not in x, stdout.split("\n"))
    stdout = chain(map(lambda x: filter(None, x), (i.split(" ") for i in stdout)))

    return [list(i)[7] for i in stdout]


def to_file(versioned_symbols):
    path = f"{os.path.dirname(os.path.realpath(__file__))}/new-gcc_s-symbols"
    with open(path, "w") as f:
        f.write(
            "Do not check this version script in: you should instead work "
            "out which symbols are missing in `lib/gcc_s.ver` and then "
            "integrate them into `lib/gcc_s.ver`. For more information, "
            "please see `doc/LLVMLibgcc.rst`.\n"
        )
        for version, symbols in versioned_symbols:
            f.write(f"{version} {{\n")
            for i in symbols:
                f.write(f"  {i};\n")
            f.write("};\n\n")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler_rt",
        type=str,
        help="Path to `libclang_rt.builtins-${ARCH}.a`.",
        required=True,
    )
    parser.add_argument(
        "--libunwind", type=str, help="Path to `libunwind.a`.", required=True
    )
    parser.add_argument(
        "--libgcc_s",
        type=str,
        help="Path to `libgcc_s.so.1`. Note that unlike the other two arguments, this is a dynamic library.",
        required=True,
    )
    return parser.parse_args()


def main():
    args = read_args()
    llvm = find_function_names(args.compiler_rt) + find_function_names(args.libunwind)
    gcc = find_function_names(args.libgcc_s)
    versioned_symbols = intersection(llvm, gcc)
    # TODO(cjdb): work out a way to integrate new symbols in with the existing
    #             ones
    to_file(versioned_symbols)


if __name__ == "__main__":
    main()
