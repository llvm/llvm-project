#!/usr/bin/env python3
"""Library kernels with realistic launch inputs, for agx-bench.

agx-bench costs a kernel with the uniforms its file's `; agx-cost:` line gives; ZLUDA's agx_libs kernels (compiled
from MLIR, without such lines) otherwise run with every argument 0, which measures their prologue only. This writes
one file per kernel listed in a shapes file (lib_shapes.txt), extracted from the compiled library IR, with an
`; agx-cost:` line that sets the kernel's arguments:

  lib_shapes.py --mlir-dir ZLUDA/agx_libs/kernels --llvm-extract BIN/llvm-extract --shapes lib_shapes.txt \\
      -o OUT_DIR LIB.opt.ll...

Arguments are named as in the MLIR source (`%N_` is `N_`); their widths come from the IR (a pointer or a 64-bit
index two 32-bit uniform words, a 32-bit index, i32 or f32 one), so a build with 32-bit indices (kernel names ending
in `_i32`) takes the same shape line. Unnamed arguments are 0; f32 arguments named alpha are 1.
"""

import argparse
import os
import re
import struct
import subprocess
import sys


def mlir_args(mlir_dir):
    """kernel -> its argument names, from every gpu.func in the MLIR sources."""
    out = {}
    for f in sorted(os.listdir(mlir_dir)):
        if not f.endswith(".mlir"):
            continue
        text = open(os.path.join(mlir_dir, f)).read()
        for m in re.finditer(r"gpu\.func @(\w+)\((.*?)\)\s*kernel", text, re.S):
            out[m.group(1)] = re.findall(r"%(\w+)\s*:", m.group(2))
    return out


def ir_widths(ir, name):
    """The 32-bit uniform words of each argument of kernel `name` in the IR text, and whether it is a float."""
    m = re.search(r"^define[^\n]*@%s\(" % re.escape(name), ir, re.M)
    if not m:
        return None
    params, depth, cur = [], 0, ""
    for ch in ir[m.end():]:
        if ch == ")" and depth == 0:
            break
        if ch == "," and depth == 0:
            params.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    params.append(cur)
    out = []
    for p in params:
        p = p.strip()
        byref = re.search(r"byref\((\w+)\)", p)
        ty = byref.group(1) if byref else p.split()[0]
        if ty == "ptr":
            out.append((2, False))
        elif ty in ("i64", "double"):
            out.append((2, ty == "double"))
        else:
            out.append((1, ty in ("float", "half")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir-dir", required=True)
    ap.add_argument("--llvm-extract", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("-o", dest="out", required=True)
    ap.add_argument("libs", nargs="+")
    a = ap.parse_args()
    names = mlir_args(a.mlir_dir)
    shapes = []
    for line in open(a.shapes):
        line = line.split("#")[0].strip()
        if line:
            kernel, rest = line.split(None, 1)
            launch = re.findall(r"--(?:wg|wg-id|grid) \S+", rest)
            vals = dict(kv.split("=") for kv in re.sub(r"--(?:wg|wg-id|grid) \S+", "", rest).split())
            shapes.append((kernel, vals, " ".join(launch)))
    os.makedirs(a.out, exist_ok=True)
    written = 0
    for lib in a.libs:
        ir = open(lib).read()
        for kernel, vals, launch in shapes:
            for name in (kernel, kernel + "_i32"):
                widths = ir_widths(ir, name)
                if widths is None:
                    continue
                args = names.get(name) or names.get(kernel)
                if not args or len(args) != len(widths):
                    sys.exit("%s: %d MLIR arguments, %d in the IR" % (name, len(args or []), len(widths)))
                words, idx = [], 0
                for arg, (n, is_float) in zip(args, widths):
                    v = vals.get(arg, vals.get(arg.rstrip("_"), "1.0" if is_float and arg == "alpha" else "0"))
                    if is_float:
                        bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
                        words.append((idx, bits))
                    else:
                        iv = int(v) & 0xFFFFFFFFFFFFFFFF
                        words.append((idx, iv & 0xFFFFFFFF))
                        if n == 2 and iv >> 32:
                            words.append((idx + 1, iv >> 32))
                    idx += n
                header = "; agx-cost: " + " ".join("--uniform %d=%d" % w for w in words if w[1]) + " " + launch
                out = os.path.join(a.out, "%s.ll" % name)
                subprocess.run([a.llvm_extract, "--func=" + name, "-S", lib, "-o", out + ".tmp"], check=True)
                open(out, "w").write(header + "\n" + open(out + ".tmp").read())
                os.remove(out + ".tmp")
                written += 1
    print("%d kernels" % written)


if __name__ == "__main__":
    main()
