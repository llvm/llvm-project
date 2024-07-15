# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is a helper to verify canonicalization patterns using Alive2
# https://alive2.llvm.org/ce/.
# It performs the following steps:
# - Filters out the provided test functions.
# - Runs the canonicalization pass on the remaining functions.
# - Lowers both the original and the canonicalized functions to LLVM IR.
# - Prints the canonicalized and the original functions side-by-side in a format
#   that can be copied into Alive2 for verification.
# Example: `python verify_canon.py canonicalize.mlir -f func1 func2 func3`

import subprocess
import tempfile
import sys
from pathlib import Path
from argparse import ArgumentParser


def filter_funcs(ir, funcs):
    if not funcs:
        return ir

    funcs_str = ",".join(funcs)
    return subprocess.check_output(
        ["mlir-opt", f"--symbol-privatize=exclude={funcs_str}", "--symbol-dce"],
        input=ir,
    )


def add_func_prefix(src, prefix):
    return src.replace("@", "@" + prefix)


def merge_ir(chunks):
    files = []
    for chunk in chunks:
        tmp = tempfile.NamedTemporaryFile(suffix=".ll")
        tmp.write(chunk)
        tmp.flush()
        files.append(tmp)

    return subprocess.check_output(["llvm-link", "-S"] + [f.name for f in files])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-f", "--func-names", nargs="+", default=[])
    args = parser.parse_args()

    file = args.file
    funcs = args.func_names

    orig_ir = Path(file).read_bytes()
    orig_ir = filter_funcs(orig_ir, funcs)

    to_llvm_args = ["--convert-to-llvm"]
    orig_args = ["mlir-opt"] + to_llvm_args
    canon_args = ["mlir-opt", "-canonicalize"] + to_llvm_args
    translate_args = ["mlir-translate", "-mlir-to-llvmir"]

    orig = subprocess.check_output(orig_args, input=orig_ir)
    canonicalized = subprocess.check_output(canon_args, input=orig_ir)

    orig = subprocess.check_output(translate_args, input=orig)
    canonicalized = subprocess.check_output(translate_args, input=canonicalized)

    enc = "utf-8"
    orig = bytes(add_func_prefix(orig.decode(enc), "src_"), enc)
    canonicalized = bytes(add_func_prefix(canonicalized.decode(enc), "tgt_"), enc)

    res = merge_ir([orig, canonicalized])

    print(res.decode(enc))
