#!/usr/bin/env python3
"""Checked mode through ZLUDA: PTX kernels compiled by ZLUDA's zoc with --agx-checked, run on the AGX Performance
Simulator's agx_run through run_checked.py (which passes the hidden arguments as ZLUDA's runtime does), with the
error record compared with what the access should give. Correct kernels must report nothing and compute the same
results as unchecked builds; unchecked builds must not report the bugs.

  zluda_tests.py --zoc ZOC --llvm-build BUILD --agx-run AGX_RUN [-k NAME]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_checked as rc  # noqa: E402

HEADER = """
.version 7.0
.target sm_70
.address_size 64
"""

N = 32

# name: (PTX, args, grid, wg, expectation, options); expectation None means no error
KERNELS = {}


def kernel(name, ptx, args, grid, wg, expect, **kw):
    KERNELS[name] = (HEADER + ptx, args, grid, wg, expect, kw)


def words(n, f=lambda i: i):
    return struct.pack("<%dI" % n, *[f(i) for i in range(n)])


def u32s(b):
    return struct.unpack("<%dI" % (len(b) // 4), b)


# out[tid] = in[tid] * 2 for tid < n
COPY = """
.visible .entry k(.param .u64 in, .param .u64 out, .param .u32 n)
{
  .reg .pred %p;
  .reg .b32 %r<5>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [in];
  ld.param.u64 %rd2, [out];
  ld.param.u32 %r1, [n];
  mov.u32 %r2, %tid.x;
  setp.ge.u32 %p, %r2, %r1;
  @%p bra done;
  cvta.to.global.u64 %rd3, %rd1;
  cvta.to.global.u64 %rd4, %rd2;
  mul.wide.u32 %rd5, %r2, 4;
  add.s64 %rd6, %rd3, %rd5;
  ld.global.u32 %r3, [%rd6];
  shl.b32 %r4, %r3, 1;
  add.s64 %rd7, %rd4, %rd5;
  st.global.u32 [%rd7], %r4;
done:
  ret;
}
"""

kernel("copy-in-bounds", COPY, lambda: [("buf", words(N)), ("buf", bytes(4 * N)), ("i32", N)], (N, 1, 1), (N, 1, 1),
       None, check=lambda r: u32s(r.buffers[1]) == tuple(2 * i for i in range(N)))
# the guard lets one thread too many through
kernel("off-by-one-store", COPY, lambda: [("buf", words(N + 1)), ("buf", bytes(4 * N)), ("i32", N + 1)],
       (N + 1, 1, 1), (N + 1, 1, 1), dict(kind=1, store=True, size=4, thread=(N, 0, 0)),
       check=lambda r: r.addr == rc.BUFFER_VA * 2 + 4 * N)
kernel("null-out", COPY, lambda: [("buf", words(N)), ("ptr", 0), ("i32", N)], (N, 1, 1), (N, 1, 1),
       dict(kind=2, store=True, size=4))

# a misaligned pointer: in + 2
kernel("misaligned-load", COPY, lambda: [("ptr", rc.BUFFER_VA + 2), ("buf", bytes(4 * N)), ("i32", N)],
       (N, 1, 1), (N, 1, 1), dict(kind=3, store=False, size=4), allocs_first=[words(N + 1)])

SHARED = """
.visible .entry k(.param .u64 out)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  .shared .align 4 .b8 s[128];
  ld.param.u64 %rd1, [out];
  mov.u32 %r1, %tid.x;
  shl.b32 %r2, %r1, 2;
  mov.u32 %r3, s;
  add.u32 %r3, %r3, %r2;
  st.shared.u32 [%r3], %r1;
  bar.sync 0;
  ld.shared.u32 %r3, [%r3];
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd2, %rd2, %rd3;
  st.global.u32 [%rd2], %r3;
  ret;
}
"""

kernel("shared-in-bounds", SHARED, lambda: [("buf", bytes(4 * N))], (N, 1, 1), (N, 1, 1), None,
       check=lambda r: u32s(r.buffers[0]) == tuple(range(N)))
kernel("shared-out-of-bounds", SHARED, lambda: [("buf", bytes(8 * N))], (2 * N, 1, 1), (2 * N, 1, 1),
       dict(kind=4, store=True, size=4), check=lambda r: r.thread[0] >= N)

DYNAMIC = """
.extern .shared .align 4 .b8 d[];
.visible .entry k(.param .u64 out)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [out];
  mov.u32 %r1, %tid.x;
  shl.b32 %r2, %r1, 2;
  mov.u32 %r3, d;
  add.u32 %r3, %r3, %r2;
  st.shared.u32 [%r3], %r1;
  ret;
}
"""

kernel("dynamic-shared-in-bounds", DYNAMIC, lambda: [("buf", bytes(4))], (N, 1, 1), (N, 1, 1), None,
       dynamic_shared=4 * N)
kernel("dynamic-shared-out-of-bounds", DYNAMIC, lambda: [("buf", bytes(4))], (N, 1, 1), (N, 1, 1),
       dict(kind=4, store=True, size=4), dynamic_shared=2 * N, check=lambda r: r.thread[0] >= N // 2)

ATOMIC = """
.visible .entry k(.param .u64 out)
{
  .reg .b32 %r<3>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [out];
  mov.u32 %r1, %tid.x;
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd2, %rd2, %rd3;
  atom.global.add.u32 %r2, [%rd2], 1;
  ret;
}
"""

kernel("atomic-in-bounds", ATOMIC, lambda: [("buf", bytes(4 * N))], (N, 1, 1), (N, 1, 1), None,
       check=lambda r: u32s(r.buffers[0]) == (1,) * N)
kernel("atomic-out-of-bounds", ATOMIC, lambda: [("buf", bytes(4 * (N - 1)))], (N, 1, 1), (N, 1, 1),
       dict(kind=1, atomic=True, size=4, thread=(N - 1, 0, 0)))

# a pointer inside a by-value struct parameter: s = { u32 n, u64 out }
STRUCT = """
.visible .entry k(.param .align 8 .b8 s[16])
{
  .reg .pred %p;
  .reg .b32 %r<3>;
  .reg .b64 %rd<4>;
  ld.param.u32 %r1, [s];
  ld.param.u64 %rd1, [s+8];
  mov.u32 %r2, %tid.x;
  setp.gt.u32 %p, %r2, %r1;
  @%p bra done;
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.u32 %rd3, %r2, 4;
  add.s64 %rd2, %rd2, %rd3;
  st.global.u32 [%rd2], %r2;
done:
  ret;
}
"""

kernel("struct-pointer-in-bounds", STRUCT, lambda: [("param", [("i32", N - 1), ("i32", 0), ("buf", bytes(4 * N))])],
       (N, 1, 1), (N, 1, 1), None, check=lambda r: u32s(r.buffers[0]) == tuple(range(N)))
# the guard is <= n rather than < n
kernel("struct-pointer-off-by-one", STRUCT, lambda: [("param", [("i32", N), ("i32", 0), ("buf", bytes(4 * N))])],
       (N + 1, 1, 1), (N + 1, 1, 1), dict(kind=1, store=True, thread=(N, 0, 0)))

# COPY over the grid
GRID_COPY = COPY.replace("  mov.u32 %r2, %tid.x;\n", """  mov.u32 %r2, %tid.x;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mad.lo.s32 %r2, %r3, %r4, %r2;
""")

# a later block reads past the end: the record names the block
kernel("second-block", GRID_COPY, lambda: [("buf", words(N)), ("buf", bytes(8 * N)), ("i32", 2 * N)], (2 * N, 1, 1),
       (N, 1, 1), dict(kind=1, store=False, threadgroup=(1, 0, 0)),
       check=lambda r: r.addr == rc.BUFFER_VA + 4 * N)
kernel("grid-in-bounds", GRID_COPY, lambda: [("buf", words(4 * N)), ("buf", bytes(16 * N)), ("i32", 4 * N)],
       (4 * N, 1, 1), (N, 1, 1), None, check=lambda r: u32s(r.buffers[1]) == tuple(2 * i for i in range(4 * N)))


# .file and .loc: the report names the store in the inlined function and the call in the kernel
INLINED = """
.file 1 "/work/inl.cu"
.file 2 "/work/put.h"
.func put(.param .u64 p, .param .u32 v)
{
  .reg .b32 %r<2>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  .loc 2 3 7
  st.global.u32 [%rd1], %r1;
  ret;
}
.visible .entry k(.param .u64 out)
{
  .reg .b32 %r<3>;
  .reg .b64 %rd<5>;
  ld.param.u64 %rd1, [out];
  mov.u32 %r1, %tid.x;
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd4, %rd2, %rd3;
  .loc 1 12 5
  {
  .param .b64 a0;
  st.param.b64 [a0], %rd4;
  .param .b32 a1;
  st.param.b32 [a1], %r1;
  call put, (a0, a1);
  }
  ret;
}
"""

kernel("source-location", INLINED, lambda: [("buf", bytes(4 * (N - 1)))], (N, 1, 1), (N, 1, 1),
       dict(kind=1, store=True, thread=(N - 1, 0, 0)),
       check=lambda r: r.site_text() == "store 4\tput put.h:3:7\tk inl.cu:12:5")


# out[tid] = *ptrs[tid]: pointers loaded from memory, checked against the allocation table
CHASE = """
.visible .entry k(.param .u64 ptrs, .param .u64 out)
{
  .reg .b32 %r<3>;
  .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [ptrs];
  ld.param.u64 %rd2, [out];
  mov.u32 %r1, %tid.x;
  cvta.to.global.u64 %rd3, %rd1;
  mul.wide.u32 %rd4, %r1, 8;
  add.s64 %rd5, %rd3, %rd4;
  ld.global.u64 %rd6, [%rd5];
  ld.global.u32 %r2, [%rd6];
  cvta.to.global.u64 %rd7, %rd2;
  mul.wide.u32 %rd4, %r1, 4;
  add.s64 %rd7, %rd7, %rd4;
  st.global.u32 [%rd7], %r2;
  ret;
}
"""


def chase_args(last):
    # data (N words) is mapped first, at BUFFER_VA; the last thread's pointer is data + 4 * last
    ptrs = b"".join(struct.pack("<Q", rc.BUFFER_VA + 4 * (t if t < N - 1 else last)) for t in range(N))
    return [("buf", ptrs), ("buf", bytes(4 * N))]


kernel("pointer-chase-in-bounds", CHASE, lambda: chase_args(N - 1), (N, 1, 1), (N, 1, 1), None,
       check=lambda r: u32s(r.buffers[2]) == tuple(100 + t for t in range(N)), allocs_first=[words(N, lambda i: 100 + i)])
kernel("pointer-chase-past-end", CHASE, lambda: chase_args(N), (N, 1, 1), (N, 1, 1),
       dict(kind=1, store=False, size=4, thread=(N - 1, 0, 0), addr=rc.BUFFER_VA + 4 * N),
       allocs_first=[words(N, lambda i: 100 + i)])


def compile_ptx(name, ptx, zoc, cpu, strict, checked, tmp):
    d = os.path.join(tmp, "%s-%s-%d-%d" % (name, cpu, strict, checked))
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "k.ptx")
    open(src, "w").write(ptx)
    cmd = [zoc, "--target", "agx", "--arch", cpu, "--output-dir", d, src]
    cmd += ["--agx-strict"] if strict else []
    cmd += ["--agx-checked"] if checked else []
    r = subprocess.run(cmd, capture_output=True, text=True)
    obj = os.path.join(d, "k.o")
    if r.returncode or not os.path.exists(obj):
        raise RuntimeError("zoc failed: " + r.stderr[-800:])
    return obj


def run_one(name, a, cpu, strict, tmp, checked=True):
    ptx, args, grid, wg, expect, kw = KERNELS[name]
    obj = compile_ptx(name, ptx, a.zoc, cpu, strict, checked, tmp)
    b = os.path.join(a.llvm_build, "bin")
    built = args()
    if "allocs_first" in kw:
        # buffers mapped before the arguments' (so a pointer argument can point into them)
        built = [("param_alloc", x) for x in kw["allocs_first"]] + built
    r = launch(obj, built, grid, wg, a.agx_run, os.path.join(b, "llvm-objcopy"), os.path.join(b, "llvm-readelf"),
               "g13" if strict else None, kw.get("dynamic_shared", 0))
    problems = []
    if expect is None:
        if r.claimed:
            problems.append("unexpected error: %s" % r)
    elif not r.claimed:
        problems.append("no error recorded")
    else:
        for f, v in expect.items():
            if getattr(r, f) != v:
                problems.append("%s = %r, expected %r (%s)" % (f, getattr(r, f), v, r))
    if "check" in kw and not problems and not kw["check"](r):
        problems.append("check failed: %s" % r)
    return problems, r


def launch(obj, args, grid, wg, agx_run, objcopy, readelf, strict, dynamic_shared):
    # leading ("param_alloc", bytes) entries only map a buffer
    pre = [v for k, v in args if k == "param_alloc"]
    args = [x for x in args if x[0] != "param_alloc"]
    return rc.launch(obj, "k", args, grid, wg, agx_run, objcopy, readelf, strict=strict,
                     dynamic_shared=dynamic_shared, pre_buffers=pre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoc", required=True)
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
        if a.k:
            print("%d failures" % fails)
            return 1 if fails else 0
        # Controls: unchecked builds do not report the bugs (so the passes above are not vacuous), and correct
        # kernels compute the same outputs as their checked builds.
        for name in ("off-by-one-store", "atomic-out-of-bounds", "shared-out-of-bounds"):
            try:
                problems, r = run_one(name, a, "agx-g14c", False, tmp, checked=False)
                unreported, how = r.claimed == 0, "reports nothing"
            except RuntimeError as e:
                unreported, how = "unmapped" in str(e), "is stopped by agx_run, not reported (%s)" % str(e)[-60:]
            if unreported:
                print("ok   control: %s unchecked %s" % (name, how))
            else:
                fails += 1
                print("FAIL control: %s unchecked: %s" % (name, r))
        for name in ("copy-in-bounds", "shared-in-bounds", "atomic-in-bounds", "struct-pointer-in-bounds"):
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
