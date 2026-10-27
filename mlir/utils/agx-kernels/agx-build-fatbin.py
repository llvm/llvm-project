#!/usr/bin/env python3
"""Compile a library kernel source (a gpu.module in standard dialects and agx operations) for several AGX chips, and
pack the objects in one fat AGX object that a runtime loads with cuModuleLoadData (ZLUDA's zluda_agx picks the entry
for its device).

The kernel source names no chip. For each chip the pipeline is:

  mlir-opt --agxir-attach-target=chip=CHIP --convert-gpu-to-agxir=use-bare-ptr-memref-call-conv=1
           --reconcile-unrealized-casts
  mlir-translate --mlir-to-llvmir   (the functions of the gpu.module)
  opt -mtriple=agx -mcpu=CHIP -O3
  llc -mtriple=agx -mcpu=CHIP -O3 -filetype=obj

With --agx-strict (the default), each chip also gets an entry named "CHIP+agx-strict", compiled with the target feature
+agx-strict in #agxir.target (so the agx operations take their portable forms) and in opt / llc (so only the
instructions Mesa's asahi compiler emits are used). A runtime in strict mode loads those entries.

With --via-air, the kernel source is compiled through AIR (standard dialects and air operations; agx operations are
not accepted), the same for every chip up to AGX's code generation:

  mlir-opt --arith-emulate-f64 --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1   (in the gpu.module)
           --reconcile-unrealized-casts
  mlir-translate --mlir-to-llvmir   (the functions of the gpu.module)
  opt -mtriple=air -passes=air-portable            (AIR's portable passes)
  opt -mtriple=agx -mcpu=CHIP -O3                  (agx-lower-air first)
  llc -mtriple=agx -mcpu=CHIP -O3 -filetype=obj

Fat AGX object layout (little-endian):

  0   8 bytes  magic "ZAGXFAT1"
  8   u32      number of entries
  12  u32      0
  16  entries, 48 bytes each:
        32 bytes  chip name (an LLVM AGX -mcpu, NUL padded)
        u64       offset of the object from the start of the file
        u64       size of the object
      then the objects (AGX ELF files), each 16-byte aligned.

  agx-build-fatbin.py --mlir-opt MLIR_OPT --mlir-translate MLIR_TRANSLATE --llc LLC [--opt OPT]
                      [--chips agx-g13g,agx-g14c,...] [--no-agx-strict] [--via-air] kernels.mlir -o out.agxfat
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


STRICT = "+agx-strict"


def portable_air(a, src, tmp):
    """The kernel source as AIR, after the AIR target's portable passes (bitcode)."""
    lowered = os.path.join(tmp, "air.mlir")
    # AIR has no double precision: f64 is emulated (arith-emulate-f64).
    subprocess.run([a.mlir_opt, src, "--pass-pipeline=builtin.module(gpu.module(arith-emulate-f64,"
                    "convert-gpu-to-air{use-bare-ptr-memref-call-conv=1}),reconcile-unrealized-casts)",
                    "-o", lowered], check=True)
    body = os.path.join(tmp, "air.body.mlir")
    open(body, "w").write(gpu_module_body(lowered))
    ll = os.path.join(tmp, "air.ll")
    subprocess.run([a.mlir_translate, "--mlir-to-llvmir", body, "-o", ll], check=True)
    portable = os.path.join(tmp, "air.portable.bc")
    subprocess.run([a.opt, "-mtriple=air", "-passes=air-portable", ll, "-o", portable], check=True)
    return portable


def compile_for(a, src, chip, tmp, features=""):
    name = chip + features
    mattr = ["-mattr=" + features] if features else []
    if a.via_air:
        optimized = os.path.join(tmp, name + ".opt.bc")
        subprocess.run([a.opt, "-mtriple=agx", "-mcpu=" + chip, *mattr, "-O3", a.portable, "-o", optimized],
                       check=True)
        obj = os.path.join(tmp, name + ".o")
        subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + chip, *mattr, "-O3", "-filetype=obj", optimized, "-o",
                        obj], check=True)
        return open(obj, "rb").read()
    attach = "--agxir-attach-target=chip=" + chip + (" features=" + features if features else "")
    lowered = os.path.join(tmp, name + ".mlir")
    subprocess.run([a.mlir_opt, src, attach,
                    "--convert-gpu-to-agxir=use-bare-ptr-memref-call-conv=1", "--reconcile-unrealized-casts",
                    "-o", lowered], check=True)
    body = os.path.join(tmp, name + ".body.mlir")
    open(body, "w").write(gpu_module_body(lowered))
    ll = os.path.join(tmp, name + ".ll")
    subprocess.run([a.mlir_translate, "--mlir-to-llvmir", body, "-o", ll], check=True)
    optimized = os.path.join(tmp, name + ".opt.ll")
    subprocess.run([a.opt, "-mtriple=agx", "-mcpu=" + chip, *mattr, "-O3", ll, "-o", optimized], check=True)
    ll = optimized
    obj = os.path.join(tmp, name + ".o")
    subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + chip, *mattr, "-O3", "-filetype=obj", ll, "-o", obj],
                   check=True)
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
    ap.add_argument("--opt", default=None, help="LLVM opt (default: next to llc)")
    ap.add_argument("--chips", default=",".join(DEFAULT_CHIPS))
    ap.add_argument("--no-agx-strict", dest="agx_strict", action="store_false",
                    help="leave out the CHIP+agx-strict entries")
    ap.add_argument("--via-air", action="store_true",
                    help="compile through AIR (convert-gpu-to-air, air-portable, agx-lower-air)")
    ap.add_argument("-o", dest="out", required=True)
    ap.add_argument("source")
    a = ap.parse_args()
    a.opt = a.opt or os.path.join(os.path.dirname(a.llc), "opt")
    with tempfile.TemporaryDirectory() as tmp:
        chips = [c for c in a.chips.split(",") if c]
        if a.via_air:
            a.portable = portable_air(a, a.source, tmp)
        objects = [(c, compile_for(a, a.source, c, tmp)) for c in chips]
        if a.agx_strict:
            # Mesa compiles for G13 and G14 only.
            objects += [(c + STRICT, compile_for(a, a.source, c, tmp, STRICT)) for c in chips
                        if c.startswith(("agx-g13", "agx-g14"))]
    open(a.out, "wb").write(pack(objects))
    return 0


if __name__ == "__main__":
    sys.exit(main())
