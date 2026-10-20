#!/usr/bin/env python3
"""Benchmark the code the LLVM AGX backend generates.

For every kernel of the suite and every chip, compile with llc, then record:
  - bytes of code (without the trailing trap padding);
  - GPRs and uniform halves, which decide occupancy;
  - waits (each one stalls the SIMD-group);
  - the estimated cycles of one SIMD-group from the AGX Performance Simulator's agx_cost.

The suite is the .ll files in kernels/ and, with --ptx, ZLUDA PTX files compiled by zoc. Results are compared with
baseline.json; a metric that gets worse by more than --tolerance is a regression and makes the script exit with 1.
--update rewrites the baseline. The JSON lines on stdout can be collected over time.

  agx-bench.py --llc build/bin/llc --cost path/to/agx_cost [--objcopy build/bin/llvm-objcopy]
               [--zoc ZLUDA/target/debug/zoc --ptx a.ptx b.ptx] [--update]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHIPS = ["agx-g13g", "agx-g14c"]
# Higher is worse for every metric.
METRICS = ["bytes", "gprs", "uniform_halves", "waits", "cycles"]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=kw.pop("text", True), **kw)


def kernel_metrics(asm, name):
    """Per-kernel numbers from the assembly: resources and instruction counts."""
    out = {}
    body = re.search(r"^%s:.*?\n(.*?)^\.Lfunc_end" % re.escape(name), asm, re.S | re.M)
    text = body.group(1) if body else ""
    m = re.search(r"uniform registers: (\d+)", text)
    out["uniform_halves"] = int(m.group(1)) if m else 0
    m = re.search(r"general purpose registers: (\d+)", text)
    out["gprs"] = int(m.group(1)) if m else 0
    insts = [l.strip() for l in text.splitlines() if l.startswith("\t") and not l.strip().startswith(("//", "."))]
    insts = [i for i in insts if i != "trap"]
    out["instructions"] = len(insts)
    out["waits"] = sum(1 for i in insts if i.startswith("wait"))
    return out


def kernels_in(asm):
    return re.findall(r"^([A-Za-z_][\w.$]*):\s+// @", asm, re.M)


def text_without_padding(obj, objcopy, tmp):
    binf = os.path.join(tmp, "code.bin")
    run([objcopy, "-O", "binary", "--only-section=.text", obj, binf])
    data = open(binf, "rb").read()
    return data


def cost(costbin, code, tmp):
    p = os.path.join(tmp, "k.bin")
    open(p, "wb").write(code)
    try:
        r = run([costbin, p, "--allow-unknown"])
        j = json.loads(r.stdout.strip().splitlines()[-1])
        return j.get("cycles")
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def compile_ll(llc, src, chip, tmp):
    asm = run([llc, "-mtriple=agx", "-mcpu=" + chip, src, "-o", "-"]).stdout
    obj = os.path.join(tmp, "k.o")
    run([llc, "-mtriple=agx", "-mcpu=" + chip, "-filetype=obj", src, "-o", obj])
    return asm, obj


def compile_ptx(zoc, src, chip, tmp):
    d = tempfile.mkdtemp(dir=tmp)
    run([zoc, "--target", "agx", "--arch", chip, "--output-dir", d, src])
    stem = os.path.splitext(os.path.basename(src))[0]
    return open(os.path.join(d, stem + ".asm")).read(), os.path.join(d, stem + ".o")


def single_kernel_code(asm, obj, name, objcopy, tmp):
    """The bytes of one kernel: llvm-objcopy of .text, sliced with the symbol table."""
    code = text_without_padding(obj, objcopy, tmp)
    syms = run([objcopy.replace("objcopy", "readelf"), "-sW", obj]).stdout
    for line in syms.splitlines():
        f = line.split()
        if len(f) >= 8 and f[3] == "FUNC" and f[7] == name:
            start, size = int(f[1], 16), int(f[2])
            body = code[start:start + size]
            # drop the 8 trailing 2-byte traps
            while body.endswith(b"\x08\x00"):
                body = body[:-2]
            return body
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--cost", required=True, help="agx_cost from the AGX Performance Simulator")
    ap.add_argument("--zoc", default=None)
    ap.add_argument("--ptx", nargs="*", default=[])
    ap.add_argument("--ll", nargs="*", default=[], help="more LLVM IR files (for example MLIR library kernels)")
    ap.add_argument("--baseline", default=os.path.join(HERE, "baseline.json"))
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.02, help="relative slack before a change counts")
    a = ap.parse_args()
    objcopy = a.objcopy or os.path.join(os.path.dirname(a.llc), "llvm-objcopy")

    sources = [("ll", os.path.join(HERE, "kernels", f)) for f in sorted(os.listdir(os.path.join(HERE, "kernels")))
               if f.endswith(".ll")]
    sources += [("ll", p) for p in a.ll]
    if a.ptx:
        if not a.zoc:
            sys.exit("--ptx needs --zoc")
        sources += [("ptx", p) for p in a.ptx]

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for kind, src in sources:
            for chip in CHIPS:
                try:
                    asm, obj = (compile_ll(a.llc, src, chip, tmp) if kind == "ll" else compile_ptx(a.zoc, src, chip, tmp))
                except subprocess.CalledProcessError as e:
                    print(json.dumps({"source": os.path.basename(src), "chip": chip, "error": (e.stderr or "")[-300:]}))
                    continue
                for name in kernels_in(asm):
                    m = kernel_metrics(asm, name)
                    code = single_kernel_code(asm, obj, name, objcopy, tmp)
                    m["bytes"] = len(code)
                    m["cycles"] = cost(a.cost, code, tmp)
                    key = "%s:%s:%s" % (os.path.basename(src), name, chip)
                    results[key] = m
                    print(json.dumps(dict(key=key, **m)))

    if a.update or not os.path.exists(a.baseline):
        json.dump(results, open(a.baseline, "w"), indent=1, sort_keys=True)
        print("baseline written: %d kernels" % len(results), file=sys.stderr)
        return 0
    base = json.load(open(a.baseline))
    worse, better = [], []
    for key, m in sorted(results.items()):
        b = base.get(key)
        if not b:
            continue
        for k in METRICS:
            x, y = b.get(k), m.get(k)
            if x is None or y is None:
                continue
            if y > x * (1 + a.tolerance) and y - x >= 1:
                worse.append("%s %s %s -> %s" % (key, k, x, y))
            elif y < x * (1 - a.tolerance) and x - y >= 1:
                better.append("%s %s %s -> %s" % (key, k, x, y))
    for l in better:
        print("better: " + l, file=sys.stderr)
    for l in worse:
        print("WORSE:  " + l, file=sys.stderr)
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
