#!/usr/bin/env python3
"""Checked-mode scenarios on the AGX Performance Simulator: each kernel is compiled with -mattr=+agx-checked and run
on agx_run through run_checked.py, and the error record (or its absence) and the outputs are compared with what the
access should give. Correct kernels must report nothing and compute the same results as unchecked builds.

  checked_tests.py --llvm-build BUILD --agx-run AGX_RUN [-k NAME]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_checked as rc  # noqa: E402

N = 64

HEADER = """
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare void @llvm.agx.threadgroup.barrier()
"""

# name: (IR, args builder, grid, wg, expectation)
# expectation: None (no error) or dict of record fields; 'check' is a function on the Result for extra checks.
KERNELS = {}


def kernel(name, ir, args, grid, wg, expect, **kw):
    KERNELS[name] = (HEADER + ir, args, grid, wg, expect, kw)


def words(n, f=lambda i: i):
    return struct.pack("<%dI" % n, *[f(i) for i in range(n)])


def u32s(b):
    return struct.unpack("<%dI" % (len(b) // 4), b)


COPY = """
define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pi = getelementptr i32, ptr addrspace(1) %in, i32 %i
  %x = load i32, ptr addrspace(1) %pi, align 4
  %y = mul i32 %x, 2
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %y, ptr addrspace(1) %po, align 4
  ret void
}
"""

GUARDED = """
define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %c = icmp ult i32 %i, %n
  br i1 %c, label %body, label %done
body:
  %pi = getelementptr i32, ptr addrspace(1) %in, i32 %i
  %x = load i32, ptr addrspace(1) %pi, align 4
  %y = mul i32 %x, 2
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %y, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void
}
"""


def copy_ok(r):
    return list(u32s(r.buffers[1])) == [2 * i for i in range(N)]


kernel("copy-in-bounds", COPY, lambda: [("buf", words(N)), ("buf", words(N, lambda i: 0)), ("i32", N)],
       (N, 1, 1), (32, 1, 1), None, check=copy_ok)
kernel("guarded-extra-threads", GUARDED, lambda: [("buf", words(N)), ("buf", words(N, lambda i: 0)), ("i32", N)],
       (N + 32, 1, 1), (32, 1, 1), None, check=copy_ok)


def oob_load(r):
    # the first out-of-bounds thread is one of the last 32, reading in[N..N+31]
    base = rc.BUFFER_VA
    return base + 4 * N <= r.addr < base + 4 * (N + 32) and r.thread[0] == (r.addr - base) // 4 - N and \
        r.threadgroup[0] == 2 and r.site_text().startswith("load 4")


kernel("off-by-one-load", COPY, lambda: [("buf", words(N)), ("buf", words(N + 32, lambda i: 0)), ("i32", N)],
       (N + 32, 1, 1), (32, 1, 1), dict(kind=1, store=False, size=4), check=oob_load)


def oob_store(r):
    base = 2 * rc.BUFFER_VA
    ok_rest = list(u32s(r.buffers[1]))[:N] == [2 * i for i in range(N)]
    return base + 4 * N <= r.addr < base + 4 * (N + 32) and ok_rest


kernel("off-by-one-store", COPY, lambda: [("buf", words(N + 32)), ("buf", words(N, lambda i: 0)), ("i32", N)],
       (N + 32, 1, 1), (32, 1, 1), dict(kind=1, store=True, size=4), check=oob_store)

kernel("null-pointer", COPY, lambda: [("ptr", 0), ("buf", words(N, lambda i: 0)), ("i32", N)],
       (32, 1, 1), (32, 1, 1), dict(kind=2, store=False))

kernel("misaligned", """
define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %b = getelementptr i8, ptr addrspace(1) %in, i32 2
  %pi = getelementptr i32, ptr addrspace(1) %b, i32 %i
  %x = load i32, ptr addrspace(1) %pi, align 2
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %x, ptr addrspace(1) %po, align 4
  ret void
}
""", lambda: [("buf", words(N)), ("buf", words(N, lambda i: 0)), ("i32", N)], (32, 1, 1), (32, 1, 1),
       dict(kind=3, store=False))

kernel("atomic-out-of-bounds", """
define agx_kernel void @k(ptr addrspace(1) %ctr, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr i32, ptr addrspace(1) %ctr, i32 %i
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 monotonic
  ret void
}
""", lambda: [("buf", words(16, lambda i: 0)), ("buf", words(4)), ("i32", 0)], (32, 1, 1), (32, 1, 1),
       dict(kind=1, atomic=True, size=4))

kernel("param-field-pointer", """
define agx_kernel void @k(ptr addrspace(4) byref([16 x i8]) %params) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pf = getelementptr i8, ptr addrspace(4) %params, i64 8
  %out = load ptr addrspace(1), ptr addrspace(4) %pf, align 8
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %i, ptr addrspace(1) %po, align 4
  ret void
}
""", lambda: [("param", [("i32", 0), ("i32", 0), ("buf", words(16, lambda i: 0))])], (32, 1, 1), (32, 1, 1),
       dict(kind=1, store=True, size=4))

kernel("global-out-of-bounds", """
@g = addrspace(1) global [8 x i32] zeroinitializer
define agx_kernel void @k(ptr addrspace(1) %out) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pg = getelementptr [8 x i32], ptr addrspace(1) @g, i32 0, i32 %i
  %x = load i32, ptr addrspace(1) %pg, align 4
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %x, ptr addrspace(1) %po, align 4
  ret void
}
""", lambda: [("buf", words(32, lambda i: 0))], (32, 1, 1), (32, 1, 1), dict(kind=1, store=False))

TG = """
@tg = internal addrspace(3) global [32 x i32] undef
define agx_kernel void @k(ptr addrspace(1) %out, i32 %scale) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %j = mul i32 %t, %scale
  %p = getelementptr [32 x i32], ptr addrspace(3) @tg, i32 0, i32 %j
  store i32 %t, ptr addrspace(3) %p, align 4
  call void @llvm.agx.threadgroup.barrier()
  %q = getelementptr [32 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  %v = load i32, ptr addrspace(3) %q, align 4
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po, align 4
  ret void
}
"""
kernel("threadgroup-in-bounds", TG, lambda: [("buf", words(32, lambda i: 0)), ("i32", 1)], (32, 1, 1), (32, 1, 1),
       None, check=lambda r: list(u32s(r.buffers[0])) == list(range(32)))
kernel("threadgroup-out-of-bounds-barrier", TG, lambda: [("buf", words(32, lambda i: 0)), ("i32", 2)],
       (32, 1, 1), (32, 1, 1), dict(kind=4, store=True, size=4),
       check=lambda r: r.addr >= 128 and [u32s(r.buffers[0])[t] for t in (0, 2, 4)] == [0, 1, 2])

kernel("stack-out-of-bounds", """
define agx_kernel void @k(ptr addrspace(1) %out, i32 %n) {
  %s = alloca [8 x i32], align 4, addrspace(5)
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %j = urem i32 %i, %n
  %p = getelementptr [8 x i32], ptr addrspace(5) %s, i32 0, i32 %j
  store i32 %i, ptr addrspace(5) %p, align 4
  %v = load i32, ptr addrspace(5) %p, align 4
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %v, ptr addrspace(1) %po, align 4
  ret void
}
""", lambda: [("buf", words(32, lambda i: 0)), ("i32", 12)], (32, 1, 1), (32, 1, 1),
       dict(kind=5, store=True, size=4), check=lambda r: 32 <= r.addr < 48)


LOOP = """
define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  br label %loop
loop:
  %j = phi i32 [ 0, %entry ], [ %j1, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %j
  %x = load i32, ptr addrspace(1) %p, align 4
  %acc1 = add i32 %acc, %x
  %j1 = add i32 %j, 1
  %more = icmp ult i32 %j1, %n
  br i1 %more, label %loop, label %done
done:
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %acc1, ptr addrspace(1) %po, align 4
  ret void
}
"""
kernel("loop-in-bounds", LOOP, lambda: [("buf", words(16)), ("buf", words(32, lambda i: 0)), ("i32", 16)],
       (32, 1, 1), (32, 1, 1), None, check=lambda r: list(u32s(r.buffers[1])) == [120] * 32)
kernel("loop-runs-past-the-end", LOOP, lambda: [("buf", words(16)), ("buf", words(32, lambda i: 0)), ("i32", 17)],
       (32, 1, 1), (32, 1, 1), dict(kind=1, store=False, addr=rc.BUFFER_VA + 64, size=4),
       check=lambda r: list(u32s(r.buffers[1])) == [0] * 32)

DIVERGENT = """
define agx_kernel void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %odd = and i32 %i, 1
  %c = icmp eq i32 %odd, 0
  br i1 %c, label %even, label %oddb
even:
  %pa = getelementptr i32, ptr addrspace(1) %a, i32 %i
  store i32 %i, ptr addrspace(1) %pa, align 4
  br label %done
oddb:
  %k = mul i32 %i, %n
  %pb = getelementptr i32, ptr addrspace(1) %b, i32 %k
  store i32 %i, ptr addrspace(1) %pb, align 4
  br label %done
done:
  ret void
}
"""
kernel("divergent-in-bounds", DIVERGENT, lambda: [("buf", words(32, lambda i: 0)), ("buf", words(32, lambda i: 0)),
       ("i32", 1)], (32, 1, 1), (32, 1, 1), None,
       check=lambda r: [u32s(r.buffers[0])[t] for t in range(0, 32, 2)] == list(range(0, 32, 2)) and
       [u32s(r.buffers[1])[t] for t in range(1, 32, 2)] == list(range(1, 32, 2)))
kernel("divergent-odd-lanes-fail", DIVERGENT, lambda: [("buf", words(32, lambda i: 0)), ("buf", words(32, lambda i: 0)),
       ("i32", 3)], (32, 1, 1), (32, 1, 1), dict(kind=1, store=True),
       check=lambda r: r.thread[0] % 2 == 1 and
       [u32s(r.buffers[0])[t] for t in range(0, 32, 2)] == list(range(0, 32, 2)))

kernel("generic-pointer", """
define agx_kernel void @k(ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %g = addrspacecast ptr addrspace(1) %out to ptr
  %p = getelementptr i32, ptr %g, i32 %i
  store i32 %i, ptr %p, align 4
  ret void
}
""", lambda: [("buf", words(16, lambda i: 0)), ("i32", 0)], (32, 1, 1), (32, 1, 1), dict(kind=1, store=True),
       check=lambda r: list(u32s(r.buffers[0])) == list(range(16)))

MANY = "define agx_kernel void @k(" + ", ".join("ptr addrspace(1) %%p%d" % j for j in range(18)) + """) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
""" + "".join("""  %%q%d = getelementptr i32, ptr addrspace(1) %%p%d, i32 %%i
  store i32 %%i, ptr addrspace(1) %%q%d, align 4
""" % (j, j, j) for j in range(18)) + """  ret void
}
"""
kernel("eighteen-pointers-in-bounds", MANY, lambda: [("buf", words(32, lambda i: 0)) for _ in range(18)],
       (32, 1, 1), (32, 1, 1), None,
       check=lambda r: all(list(u32s(b)) == list(range(32)) for b in r.buffers))


MIXED = """
define agx_kernel void @k(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %skew) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %row = mul i32 %i, 16
  %b0 = getelementptr i8, ptr addrspace(1) %buf, i32 %skew
  %b = getelementptr i8, ptr addrspace(1) %b0, i32 %row
  %p4 = getelementptr i8, ptr addrspace(1) %b, i32 4
  %x = load i32, ptr addrspace(1) %p4, align 4
  %p8 = getelementptr i8, ptr addrspace(1) %b, i32 8
  %y = load i64, ptr addrspace(1) %p8, align 8
  %y32 = trunc i64 %y to i32
  %s = add i32 %x, %y32
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %s, ptr addrspace(1) %po, align 4
  ret void
}
"""
kernel("group-mixed-sizes-aligned", MIXED, lambda: [("buf", words(4 * 32 + 4, lambda i: i)),
       ("buf", words(32, lambda i: 0)), ("i32", 0)], (32, 1, 1), (32, 1, 1), None,
       check=lambda r: list(u32s(r.buffers[1])) == [(4 * i + 1) + (4 * i + 2) for i in range(32)])
kernel("group-misaligned-base", MIXED, lambda: [("buf", words(4 * 32 + 4, lambda i: i)),
       ("buf", words(32, lambda i: 0)), ("i32", 4)], (32, 1, 1), (32, 1, 1), dict(kind=3, store=False))

UNROLLED = """
define agx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %base = mul i32 %i, 4
  %p0 = getelementptr i32, ptr addrspace(1) %in, i32 %base
  %x0 = load i32, ptr addrspace(1) %p0, align 4
  %p1 = getelementptr i32, ptr addrspace(1) %p0, i32 1
  %x1 = load i32, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr i32, ptr addrspace(1) %p0, i32 2
  %x2 = load i32, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr i32, ptr addrspace(1) %p0, i32 3
  %x3 = load i32, ptr addrspace(1) %p3, align 4
  %a = add i32 %x0, %x1
  %b = add i32 %x2, %x3
  %c = add i32 %a, %b
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %c, ptr addrspace(1) %po, align 4
  ret void
}
"""
kernel("unrolled-in-bounds", UNROLLED, lambda: [("buf", words(128)), ("buf", words(32, lambda i: 0)), ("i32", 0)],
       (32, 1, 1), (32, 1, 1), None,
       check=lambda r: list(u32s(r.buffers[1])) == [16 * i + 6 for i in range(32)])
kernel("unrolled-last-element-past-the-end", UNROLLED,
       lambda: [("buf", words(127)), ("buf", words(32, lambda i: 0)), ("i32", 0)],
       (32, 1, 1), (32, 1, 1), dict(kind=1, store=False, size=16), check=lambda r: r.thread[0] == 31)

kernel("threadgroup-overflow-into-neighbour", """
@a = internal addrspace(3) global [16 x i32] undef
@b = internal addrspace(3) global [16 x i32] undef
define agx_kernel void @k(ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pb = getelementptr [16 x i32], ptr addrspace(3) @b, i32 0, i32 0
  store i32 7, ptr addrspace(3) %pb, align 4
  %p = getelementptr [16 x i32], ptr addrspace(3) @a, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %p, align 4
  ret void
}
""", lambda: [("buf", words(4))], (32, 1, 1), (32, 1, 1), dict(kind=4, store=True),
       check=lambda r: r.thread[0] >= 16)

DYN = """
@dyn = external addrspace(3) global [0 x i32]
define agx_kernel void @k(ptr addrspace(1) %out, i32 %n) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %p = getelementptr [0 x i32], ptr addrspace(3) @dyn, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %p, align 4
  %v = load i32, ptr addrspace(3) %p, align 4
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %v, ptr addrspace(1) %po, align 4
  ret void
}
"""
kernel("dynamic-shared-in-bounds", DYN, lambda: [("buf", words(32, lambda i: 0)), ("i32", 0)], (32, 1, 1),
       (32, 1, 1), None, check=lambda r: list(u32s(r.buffers[0])) == list(range(32)), dynamic_shared=128)
kernel("dynamic-shared-overflow", DYN, lambda: [("buf", words(32, lambda i: 0)), ("i32", 0)], (32, 1, 1),
       (32, 1, 1), dict(kind=4, store=True), check=lambda r: r.thread[0] >= 16, dynamic_shared=64)


def poisoned(r):
    return list(u32s(r.buffers[1])) == [0] * N


kernel("poisoned-launch-does-nothing", COPY, lambda: [("buf", words(N)), ("buf", words(N, lambda i: 0)), ("i32", N)],
       (N, 1, 1), (32, 1, 1), "poisoned", check=poisoned, claimed=1)


def run_one(name, a, cpu, strict, tmp, checked=True):
    ir, args, grid, wg, expect, kw = KERNELS[name]
    src = os.path.join(tmp, name + ".ll")
    obj = os.path.join(tmp, name + "-" + cpu + ".o")
    open(src, "w").write(ir)
    mattr = ("+agx-checked" if checked else "-agx-checked") + (",+agx-strict" if strict else "")
    obj = obj.replace(".o", "-checked.o" if checked else "-unchecked.o")
    subprocess.run([os.path.join(a.llvm_build, "bin", "llc"), "-mtriple=agx", "-mcpu=" + cpu, "-mattr=" + mattr,
                    "-filetype=obj", src, "-o", obj], check=True)
    bin_ = os.path.join(a.llvm_build, "bin")
    r = rc.launch(obj, "k", args(), grid, wg, a.agx_run, os.path.join(bin_, "llvm-objcopy"),
                  os.path.join(bin_, "llvm-readelf"), claimed=kw.get("claimed", 0),
                  dynamic_shared=kw.get("dynamic_shared", 0),
                  strict=("g13" if cpu.startswith("agx-g13") else "g14") if strict else None)
    problems = []
    if expect is None and r.claimed:
        problems.append("unexpected error: %s" % r)
    if expect == "poisoned" and r.claimed != 1:
        problems.append("the record changed")
    if isinstance(expect, dict):
        if not r.claimed:
            problems.append("no error recorded")
        for f, v in expect.items():
            if getattr(r, f) != v:
                problems.append("%s = %r, expected %r (%s)" % (f, getattr(r, f), v, r))
    if "check" in kw and not problems and not kw["check"](r):
        problems.append("check failed: %s" % r)
    return problems, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-build", required=True)
    ap.add_argument("--agx-run", required=True)
    ap.add_argument("-k", default="")
    a = ap.parse_args()
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in KERNELS:
            if a.k and a.k not in name:
                continue
            for cpu, strict in (("agx-g14c", False), ("agx-g13g", True)):
                try:
                    problems, r = run_one(name, a, cpu, strict, tmp)
                except Exception as e:  # noqa: BLE001
                    problems, r = ["%s" % e], None
                tag = "%s %s%s" % (name, cpu, " strict" if strict else "")
                if problems:
                    fails += 1
                    print("FAIL", tag, "; ".join(problems))
                else:
                    print("ok  ", tag, "|", r)
        # Controls: built without the feature, a bug is not reported (so the passes above are not vacuous), and
        # correct kernels compute the same outputs as their checked builds.
        for name in ("off-by-one-load", "null-pointer", "stack-out-of-bounds"):
            try:
                problems, r = run_one(name, a, "agx-g14c", False, tmp, checked=False)
                unreported = r.claimed == 0 and any("no error recorded" in p for p in problems)
                how = "reports nothing"
            except RuntimeError as e:
                # agx_run stops on an unmapped access, where the hardware would read 0 (soft faults)
                unreported, how = "unmapped" in str(e), "is stopped by agx_run, not reported (%s)" % str(e)[-60:]
            if unreported:
                print("ok   control: %s unchecked %s" % (name, how))
            else:
                fails += 1
                print("FAIL control: %s unchecked: %s" % (name, problems))
        for name in ("copy-in-bounds", "loop-in-bounds", "divergent-in-bounds", "threadgroup-in-bounds"):
            _, rc_ = run_one(name, a, "agx-g14c", False, tmp, checked=True)
            _, ru = run_one(name, a, "agx-g14c", False, tmp, checked=False)
            if rc_.buffers == ru.buffers:
                print("ok   control: %s checked and unchecked outputs are identical" % name)
            else:
                fails += 1
                print("FAIL control: %s outputs differ between checked and unchecked builds" % name)
    print("%d failures" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
