# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run canonicalization, convert IR to LLVM and convert to format suitable to
# verification against Alive2 https://alive2.llvm.org/ce/.
# Example: `python verify_canon.py canonicalize.mlir func1 func2 func3`

import subprocess
import tempfile
import sys
from pathlib import Path


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
    argv = sys.argv
    if len(argv) < 2:
        print(f"usage: {argv[0]} canonicalize.mlir [func1] [func2] ...")
        exit(0)

    file = argv[1]
    funcs = argv[2:]

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
