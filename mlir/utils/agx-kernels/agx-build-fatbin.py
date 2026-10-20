#!/usr/bin/env python3
"""Compile a library kernel source (a gpu.module in standard dialects and agx operations) for several AGX chips, and
pack the objects in one fat AGX object that a runtime loads with cuModuleLoadData (ZLUDA's zluda_agx picks the entry
for its device).

The kernel source names no chip. For each chip the pipeline is:

  mlir-opt --agxir-attach-target=chip=CHIP --convert-gpu-to-agxir=use-bare-ptr-memref-call-conv=1
           --reconcile-unrealized-casts
  mlir-translate --mlir-to-llvmir   (the functions of the gpu.module)
  llc -mtriple=agx -mcpu=CHIP -filetype=obj

Fat AGX object layout (little-endian):

  0   8 bytes  magic "ZAGXFAT1"
  8   u32      number of entries
  12  u32      0
  16  entries, 48 bytes each:
        32 bytes  chip name (an LLVM AGX -mcpu, NUL padded)
        u64       offset of the object from the start of the file
        u64       size of the object
      then the objects (AGX ELF files), each 16-byte aligned.

  agx-build-fatbin.py --mlir-opt MLIR_OPT --mlir-translate MLIR_TRANSLATE --llc LLC
                      [--chips agx-g13g,agx-g14c,...] kernels.mlir -o out.agxfat
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

MAGIC = b"ZAGXFAT1"
# Every chip the LLVM AGX backend generates code for.
DEFAULT_CHIPS = ["agx-g13g", "agx-g13s", "agx-g13c", "agx-g13d", "agx-g14g", "agx-g14s", "agx-g14c", "agx-g14d"]


def gpu_module_body(path):
    """The operations inside the gpu.module of a lowered MLIR file, as a builtin module."""
    lines = open(path).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("gpu.module"))
    depth = lines[start].count("{") - lines[start].count("}")
    end = start
    while depth:
        end += 1
        depth += lines[end].count("{") - lines[end].count("}")
    return "module {\n" + "\n".join(lines[start + 1:end]) + "\n}\n"


def compile_for(a, src, chip, tmp):
    lowered = os.path.join(tmp, chip + ".mlir")
    subprocess.run([a.mlir_opt, src, "--agxir-attach-target=chip=" + chip,
                    "--convert-gpu-to-agxir=use-bare-ptr-memref-call-conv=1", "--reconcile-unrealized-casts",
                    "-o", lowered], check=True)
    body = os.path.join(tmp, chip + ".body.mlir")
    open(body, "w").write(gpu_module_body(lowered))
    ll = os.path.join(tmp, chip + ".ll")
    subprocess.run([a.mlir_translate, "--mlir-to-llvmir", body, "-o", ll], check=True)
    obj = os.path.join(tmp, chip + ".o")
    subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + chip, "-O3", "-filetype=obj", ll, "-o", obj], check=True)
    return open(obj, "rb").read()


def pack(objects):
    header = MAGIC + struct.pack("<II", len(objects), 0)
    offset = len(header) + 48 * len(objects)
    entries, blobs = b"", b""
    for chip, data in objects:
        pad = (-offset) % 16
        blobs += b"\0" * pad
        offset += pad
        entries += chip.encode().ljust(32, b"\0") + struct.pack("<QQ", offset, len(data))
        blobs += data
        offset += len(data)
    return header + entries + blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir-opt", required=True)
    ap.add_argument("--mlir-translate", required=True)
    ap.add_argument("--llc", required=True)
    ap.add_argument("--chips", default=",".join(DEFAULT_CHIPS))
    ap.add_argument("-o", dest="out", required=True)
    ap.add_argument("source")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        objects = [(c, compile_for(a, a.source, c, tmp)) for c in a.chips.split(",") if c]
    open(a.out, "wb").write(pack(objects))
    return 0


if __name__ == "__main__":
    sys.exit(main())
