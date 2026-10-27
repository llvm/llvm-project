#!/usr/bin/env python3
"""Compile a library kernel source (a gpu.module in standard dialects and air operations) for several AGX chips, and
pack the objects in one fat AGX object that a runtime loads with cuModuleLoadData (ZLUDA's zluda_agx picks the entry
for its device).

The kernel source names no chip. It is compiled through AIR, the same for every chip up to AGX's code generation:

  mlir-opt --arith-emulate-f64 --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1   (in the gpu.module)
           --reconcile-unrealized-casts
  mlir-translate --mlir-to-llvmir   (the functions of the gpu.module)
  opt -mtriple=air -passes=air-portable            (AIR's portable passes)
  opt -mtriple=agx -mcpu=CHIP -O3                  (agx-lower-air first; one run shared by the G13 and G14 chips)
  llc -mtriple=agx -mcpu=CHIP -O3 -filetype=obj

With --agx-strict (the default), each G13 and G14 chip also gets an entry named "CHIP+agx-strict", compiled with the
target feature +agx-strict in opt and llc, so that only the instructions Mesa's asahi compiler emits are used. A
runtime in strict mode loads those entries.

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
                      [--chips agx-g13g,agx-g14c,...] [--no-agx-strict] kernels.mlir -o out.agxfat
"""

import argparse
import concurrent.futures
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


# AGX's IR-level target hooks (TargetTransformInfo, agx-lower-air) differ between chips only in the GPR count, which
# is 128 on G13 and G14 as on the generic CPU: those chips share one opt -O3 run per feature string, made without
# -mcpu so that no target-cpu attribute overrides llc's. Other chips get their own.
SHARED_OPT = ("agx-g13", "agx-g14")


def optimize(a, chip, tmp, features):
    mattr = ["-mattr=" + features] if features else []
    cpu = [] if chip is None else ["-mcpu=" + chip]
    optimized = os.path.join(tmp, (chip or "shared") + features + ".opt.bc")
    subprocess.run([a.opt, "-mtriple=agx", *cpu, *mattr, "-O3", a.portable, "-o", optimized], check=True)
    return optimized


def compile_for(a, chip, tmp, features, optimized):
    mattr = ["-mattr=" + features] if features else []
    obj = os.path.join(tmp, chip + features + ".o")
    subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + chip, *mattr, "-O3", "-filetype=obj", optimized, "-o", obj],
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
    # The only pipeline now; accepted for the builds that still pass it.
    ap.add_argument("--via-air", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-j", dest="jobs", type=int, default=os.cpu_count() or 1,
                    help="chips to compile at once (default: the processor count)")
    ap.add_argument("-o", dest="out", required=True)
    ap.add_argument("source")
    a = ap.parse_args()
    a.opt = a.opt or os.path.join(os.path.dirname(a.llc), "opt")
    with tempfile.TemporaryDirectory() as tmp:
        chips = [c for c in a.chips.split(",") if c]
        a.portable = portable_air(a, a.source, tmp)
        builds = [(c, "") for c in chips]
        if a.agx_strict:
            # Mesa compiles for G13 and G14 only.
            builds += [(c, STRICT) for c in chips if c.startswith(("agx-g13", "agx-g14"))]
        # The builds are independent opt and llc processes: run them side by side.
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
            opts = {}
            for c, f in builds:
                key = (None if c.startswith(SHARED_OPT) else c, f)
                if key not in opts:
                    opts[key] = pool.submit(optimize, a, key[0], tmp, f)
            objs = list(pool.map(
                lambda b: compile_for(a, b[0], tmp, b[1],
                                      opts[(None if b[0].startswith(SHARED_OPT) else b[0], b[1])].result()),
                builds))
        objects = [(c + f, o) for (c, f), o in zip(builds, objs)]
    open(a.out, "wb").write(pack(objects))
    return 0


if __name__ == "__main__":
    sys.exit(main())
