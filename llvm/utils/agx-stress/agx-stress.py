#!/usr/bin/env python3
"""Stress the LLVM AGX backend with kernels at its limits, and check the results on the AGX Performance Simulator.

Each case generates LLVM IR, compiles it with llc for every chip, runs it with agx_run and compares the output
buffer with a Python model of the kernel:

  pressure-*     160 values live at once: unbounded registers, at 1024 threads per threadgroup (52 GPRs) and at
                 "agx-max-gprs"="24", which spill to the stack;
  big-code       a loop around 12000 instructions (about 150 KB of code);
  nesting        48 nested divergent ifs;
  tg-48k         48 KiB of threadgroup memory;
  generic        a load through a generic pointer that may name threadgroup or device memory;
  generic3       the same with private memory as a third choice.

  agx-stress.py --llc build/bin/llc --run path/to/agx_run [--objcopy build/bin/llvm-objcopy] [--only NAME]

Exits with 1 if a case fails to compile or gives a wrong result.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

CHIPS = ["agx-g13g", "agx-g14c"]
M = 0xFFFFFFFF
OUT = 0x100000
IN = 0x200000


def pressure(attr):
    n = 160
    a = [(i * 2654435761) & M for i in range(n)]
    lines = ["declare i32 @llvm.agx.thread.position.in.grid.x()",
             "define agx_kernel void @k(ptr addrspace(1) %%a, ptr addrspace(1) %%out)%s {" % (" #0" if attr else ""),
             "  %t = call i32 @llvm.agx.thread.position.in.grid.x()"]
    for i in range(n):
        lines += ["  %%p%d = getelementptr i32, ptr addrspace(1) %%a, i32 %d" % (i, i),
                  "  %%l%d = load volatile i32, ptr addrspace(1) %%p%d" % (i, i),
                  "  %%v%d = add i32 %%l%d, %%t" % (i, i)]
    lines.append("  %s0 = add i32 %v0, 0")
    for i in range(1, n):
        lines += ["  %%m%d = mul i32 %%v%d, %d" % (i, n - i, i + 1), "  %%s%d = xor i32 %%s%d, %%m%d" % (i, i - 1, i)]
    lines += ["  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t",
              "  store i32 %%s%d, ptr addrspace(1) %%po" % (n - 1), "  ret void", "}"]
    if attr:
        lines.append("attributes #0 = { %s }" % attr)

    def expect(t):
        v = [(x + t) & M for x in a]
        s = v[0]
        for i in range(1, n):
            s ^= (v[n - i] * (i + 1)) & M
        return s

    return dict(ir="\n".join(lines), threads=64, wg=32, inputs=struct.pack("<%dI" % n, *a),
                uniforms=[IN, 0, OUT, 0], expect=[expect(t) for t in range(64)])


def big_code():
    n, iters = 12000, 3
    consts = [(k * 2654435761 % 1000003) | 1 for k in range(n)]
    ops = ["add", "xor", "mul"]
    lines = ["declare i32 @llvm.agx.thread.position.in.grid.x()",
             "define agx_kernel void @k(ptr addrspace(1) %out, i32 %n) {",
             "entry:", "  %t = call i32 @llvm.agx.thread.position.in.grid.x()", "  br label %loop",
             "loop:", "  %i = phi i32 [0, %entry], [%inext, %loop]",
             "  %%acc = phi i32 [%%t, %%entry], [%%a%d, %%loop]" % (n - 1)]
    prev = "%acc"
    for k in range(n):
        lines.append("  %%a%d = %s i32 %s, %d" % (k, ops[k % 3], prev, consts[k]))
        prev = "%%a%d" % k
    lines += ["  %inext = add i32 %i, 1", "  %c = icmp ult i32 %inext, %n", "  br i1 %c, label %loop, label %exit",
              "exit:", "  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t",
              "  store i32 %s, ptr addrspace(1) %%po" % prev, "  ret void", "}"]

    def expect(t):
        acc = t
        for _ in range(iters):
            for k, c in enumerate(consts):
                acc = (acc + c) & M if k % 3 == 0 else acc ^ c if k % 3 == 1 else (acc * c) & M
        return acc

    return dict(ir="\n".join(lines), threads=32, wg=32, uniforms=[OUT, 0, iters],
                expect=[expect(t) for t in range(32)])


def nesting():
    d = 48
    lines = ["declare i32 @llvm.agx.thread.position.in.grid.x()", "define agx_kernel void @k(ptr addrspace(1) %out) {",
             "b0:", "  %t = call i32 @llvm.agx.thread.position.in.grid.x()", "  %x0 = add i32 %t, 0"]
    for i in range(d):
        lines += ["  %%bit%d = and i32 %%t, %d" % (i, 1 << (i % 5)), "  %%c%d = icmp ne i32 %%bit%d, 0" % (i, i),
                  "  br i1 %%c%d, label %%in%d, label %%j%d" % (i, i, i),
                  "in%d:" % i, "  %%y%d = mul i32 %%x%d, 3" % (i, i), "  br label %%next%d" % i,
                  "next%d:" % i, "  %%x%d = add i32 %%y%d, %d" % (i + 1, i, i + 1)]
    lines.append("  br label %%j%d" % (d - 1))
    for i in reversed(range(d)):
        pred = "b0" if i == 0 else "next%d" % (i - 1)
        taken = ("%%x%d, %%next%d" % (i + 1, i)) if i == d - 1 else ("%%r%d, %%j%d" % (i + 1, i + 1))
        lines += ["j%d:" % i, "  %%r%d = phi i32 [%s], [%%x%d, %%%s]" % (i, taken, i, pred)]
        if i > 0:
            lines.append("  br label %%j%d" % (i - 1))
    lines += ["  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t", "  store i32 %r0, ptr addrspace(1) %po",
              "  ret void", "}"]

    def expect(t):
        x = t
        for i in range(d):
            if not t & (1 << (i % 5)):
                break
            x = (x * 3 + i + 1) & M
        return x

    return dict(ir="\n".join(lines), threads=32, wg=32, uniforms=[OUT, 0], expect=[expect(t) for t in range(32)])


def tg_48k():
    ir = """@buf = internal addrspace(3) global [12288 x i32] undef
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
define agx_kernel void @k(ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %i = add i32 %t, 12000
  %p = getelementptr [12288 x i32], ptr addrspace(3) @buf, i32 0, i32 %i
  store i32 %t, ptr addrspace(3) %p
  %v = load i32, ptr addrspace(3) %p
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po
  ret void
}"""
    return dict(ir=ir, threads=64, wg=64, shared=49152, uniforms=[OUT, 0], expect=list(range(64)))


def generic(sel):
    ir = """@buf = internal addrspace(3) global [64 x i32] undef
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(1) %dev, i32 %sel) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %s = getelementptr [64 x i32], ptr addrspace(3) @buf, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %s
  %gs = addrspacecast ptr addrspace(3) %s to ptr
  %gd = addrspacecast ptr addrspace(1) %dev to ptr
  %c = icmp eq i32 %sel, 0
  %p = select i1 %c, ptr %gs, ptr %gd
  %v = load i32, ptr %p
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po
  ret void
}"""
    return dict(ir=ir, threads=32, wg=32, shared=256, inputs=struct.pack("<I", 1000),
                uniforms=[OUT, 0, IN, 0, sel], expect=list(range(32)) if sel == 0 else [1000] * 32)


def generic3(sel):
    """A generic pointer to threadgroup (sel 0), private (sel 1) or device (sel 2) memory."""
    ir = """@buf = internal addrspace(3) global [64 x i32] undef
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(1) %dev, i32 %sel) {
  %priv = alloca [4 x i32], addrspace(5)
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pp = getelementptr [4 x i32], ptr addrspace(5) %priv, i32 0, i32 %sel
  %pv = add i32 %t, 100
  store volatile i32 %pv, ptr addrspace(5) %pp
  %s = getelementptr [64 x i32], ptr addrspace(3) @buf, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %s
  %gs = addrspacecast ptr addrspace(3) %s to ptr
  %gp = addrspacecast ptr addrspace(5) %pp to ptr
  %gd = addrspacecast ptr addrspace(1) %dev to ptr
  %c0 = icmp eq i32 %sel, 0
  %c1 = icmp eq i32 %sel, 1
  %p12 = select i1 %c1, ptr %gp, ptr %gd
  %p = select i1 %c0, ptr %gs, ptr %p12
  %v = load i32, ptr %p
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po
  ret void
}"""
    expect = list(range(32)) if sel == 0 else [t + 100 for t in range(32)] if sel == 1 else [1000] * 32
    return dict(ir=ir, threads=32, wg=32, shared=256, inputs=struct.pack("<I", 1000),
                uniforms=[OUT, 0, IN, 0, sel], expect=expect)


CASES = {
    "pressure": lambda: pressure(None),
    "pressure-1024-threads": lambda: pressure('"agx-max-threads-per-threadgroup"="1024"'),
    "pressure-24-gprs": lambda: pressure('"agx-max-gprs"="24"'),
    "big-code": big_code,
    "nesting": nesting,
    "tg-48k": tg_48k,
    "generic-threadgroup": lambda: generic(0),
    "generic-device": lambda: generic(1),
    "generic3-threadgroup": lambda: generic3(0),
    "generic3-private": lambda: generic3(1),
    "generic3-device": lambda: generic3(2),
}


def run_case(name, case, chip, a, tmp):
    ll, obj, code = (os.path.join(tmp, name + s) for s in (".ll", ".o", ".bin"))
    open(ll, "w").write(case["ir"] + "\n")
    subprocess.run([a.llc, "-mtriple=agx", "-mcpu=" + chip, "-filetype=obj", ll, "-o", obj], check=True,
                   capture_output=True, text=True)
    subprocess.run([a.objcopy, "-O", "binary", "--only-section=.text", obj, code], check=True)
    out = os.path.join(tmp, name + ".out")
    n = len(case["expect"])
    cmd = [a.run, code, "--grid", "%d,1,1" % case["threads"], "--wg", "%d,1,1" % case["wg"],
           "--alloc", "%#x=%d" % (OUT, 4 * n), "--dump", "%#x=%d:%s" % (OUT, 4 * n, out)]
    if case.get("shared"):
        cmd += ["--shared", str(case["shared"])]
    if case.get("inputs"):
        inp = os.path.join(tmp, name + ".in")
        open(inp, "wb").write(case["inputs"])
        cmd += ["--map", "%#x=%s" % (IN, inp)]
    for i, v in enumerate(case["uniforms"]):
        cmd += ["--uniform", "%d=%d" % (i, v)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    got = struct.unpack("<%dI" % n, open(out, "rb").read())
    return [i for i in range(n) if got[i] != case["expect"][i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llc", default="llc")
    ap.add_argument("--run", required=True, help="agx_run from the AGX Performance Simulator")
    ap.add_argument("--objcopy", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    a.objcopy = a.objcopy or os.path.join(os.path.dirname(a.llc), "llvm-objcopy")
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, make in CASES.items():
            if a.only and a.only != name:
                continue
            case = make()
            for chip in CHIPS:
                try:
                    bad = run_case(name, case, chip, a, tmp)
                    status = "ok" if not bad else "WRONG for threads %s" % bad[:8]
                except subprocess.CalledProcessError as e:
                    bad, status = True, "FAILED: " + (e.stderr or "")[-300:].strip()
                failed += bool(bad)
                print("%-24s %-9s %s" % (name, chip, status))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
