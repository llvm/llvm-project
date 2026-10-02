#!/usr/bin/env python3
"""Generates llvm/test/CodeGen/X86/win64-tailcc-unwind-*.ll.

The test files are repetitive (each case has a function under test, a runner
and the targets it tail calls), so they are generated. Edit this script and
re-run it, rather than editing the .ll files.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

GPRS = [("rbx", 0x11), ("rsi", 0x22), ("rdi", 0x33), ("r12", 0x44),
        ("r13", 0x55), ("r14", 0x66), ("r15", 0x77)]


def pat(b):
    return "0x" + ("%02x" % b) * 8


def sentinel_asm(skip):
    lines, clob = [], []
    for r, b in GPRS:
        if r in skip:
            continue
        lines.append("movabsq $$%s, %%%s" % (pat(b), r))
        clob.append("~{%s}" % r)
    for i in range(6, 16):
        lines.append("movabsq $$%s, %%rax" % pat(0x80 + i))
        lines.append("movq %%rax, %%xmm%d" % i)
        lines.append("punpcklqdq %%xmm%d, %%xmm%d" % (i, i))
        clob.append("~{xmm%d}" % i)
    clob += ["~{rax}", "~{dirflag}", "~{fpsr}", "~{flags}"]
    return "\\0A".join(lines), ",".join(clob)


class Case:
    def __init__(self, name, paths, k=3, cc="tailcc", fp=False, noop=False,
                 dynamic=False, xmm=False, async_=False, large=None,
                 huge=False, consts=False):
        self.name, self.paths, self.k, self.cc = name, paths, k, cc
        self.fp, self.noop, self.dynamic = fp, noop, dynamic
        self.xmm, self.async_, self.large, self.huge = xmm, async_, large, huge
        self.consts = consts
        # Values computed before the call so that they are in callee-saved
        # registers when stored over the return address.
        self.hoist = set()
        if noop and large is None and k <= 4:
            for kind, m in paths:
                if kind in ("int", "mem", "async"):
                    a = old_slot_arg(m)
                    if a and a > k:
                        self.hoist.add(a)


def old_slot_arg(m):
    """Index of the argument of an m-argument tail callee that is stored over
    the return address, for a caller whose own arguments are all in registers
    (home area 32 bytes, padded so that B + 8 is a multiple of 16)."""
    b = 32 + 8 * (m - 4)
    if (b + 8) % 16:
        b += 8
    first = 32 + (40 - b)
    if first > -8:
        return None
    return (-8 - first) // 8 + 1 + 4


def int_target(m):
    args = ", ".join("i64 %%a%d" % i for i in range(1, m + 1))
    body = ["define tailcc void @target_i%d(%s) {" % (m, args)]
    prev = None
    for i in range(1, m + 1):
        body.append("  %%c%d = icmp eq i64 %%a%d, %d" % (i, i, 1000 + i))
        if prev is None:
            prev = "%c1"
        else:
            body.append("  %%t%d = and i1 %s, %%c%d" % (i, prev, i))
            prev = "%%t%d" % i
    body += ["  %ok = zext i1 " + prev + " to i32",
             "  store volatile i32 %ok, ptr @args_ok", "  ret void", "}"]
    return "\n".join(body)


def dbl_target(ni, nd):
    args = ", ".join(["i64 %%a%d" % i for i in range(1, ni + 1)] +
                     ["double %%d%d" % i for i in range(1, nd + 1)])
    body = ["define tailcc void @target_d%d_%d(%s) {" % (ni, nd, args)]
    prev = None
    for i in range(1, ni + 1):
        body.append("  %%c%d = icmp eq i64 %%a%d, %d" % (i, i, 1000 + i))
        if prev is None:
            prev = "%c1"
        else:
            body.append("  %%t%d = and i1 %s, %%c%d" % (i, prev, i))
            prev = "%%t%d" % i
    for i in range(1, nd + 1):
        body.append("  %%e%d = fcmp oeq double %%d%d, 1.001000e+03" % (i, i))
        body.append("  %%u%d = and i1 %s, %%e%d" % (i, prev, i))
        prev = "%%u%d" % i
    body += ["  %ok = zext i1 " + prev + " to i32",
             "  store volatile i32 %ok, ptr @args_ok", "  ret void", "}"]
    return "\n".join(body)


def async_target(m):
    args = ", ".join(["ptr swiftasync %ctx"] +
                     ["i64 %%a%d" % i for i in range(1, m + 1)])
    body = ["define swifttailcc void @target_a%d(%s) {" % (m, args),
            "  %cx = ptrtoint ptr %ctx to i64",
            "  %c0 = icmp eq i64 %cx, 4660"]
    prev = "%c0"
    for i in range(1, m + 1):
        body.append("  %%c%d = icmp eq i64 %%a%d, %d" % (i, i, 1000 + i))
        body.append("  %%t%d = and i1 %s, %%c%d" % (i, prev, i))
        prev = "%%t%d" % i
    body += ["  %ok = zext i1 " + prev + " to i32",
             "  store volatile i32 %ok, ptr @args_ok", "  ret void", "}"]
    return "\n".join(body)


HUGE = 600
BIG = 9000


def huge_target():
    return "\n".join([
        "define tailcc void @target_huge([%d x i64] %%x) {" % HUGE,
        "  %%a = extractvalue [%d x i64] %%x, 0" % HUGE,
        "  %%b = extractvalue [%d x i64] %%x, %d" % (HUGE, HUGE - 1),
        "  %c = icmp eq i64 %a, 0", "  %d = icmp eq i64 %b, 0",
        "  %t = and i1 %c, %d", "  %ok = zext i1 %t to i32",
        "  store volatile i32 %ok, ptr @args_ok", "  ret void", "}"])


def int_args(case, m, prefix):
    """Argument list for an integer target with m arguments; defines values
    for the ones past the incoming ones. Returns (defs, list)."""
    defs, args = [], []
    for i in range(1, m + 1):
        if case.large is not None:
            args.append("i64 %d" % (1000 + i))
        elif i <= case.k:
            args.append("i64 %%p%d" % i)
        elif i in case.hoist:
            args.append("i64 %%hold%d" % i)
        elif case.consts:
            args.append("i64 %d" % (1000 + i))
        else:
            defs.append("  %%%s%d = add i64 %%p1, %d" % (prefix, i, i - 1))
            args.append("i64 %%%s%d" % (prefix, i))
    return defs, args


def test_function(case):
    cc = case.cc
    if case.large is not None:
        params = "i64 %%sel, [%d x i64] %%big" % case.large
    else:
        params = ", ".join(["i64 %sel"] +
                           ["i64 %%p%d" % i for i in range(1, case.k + 1)])
    if case.async_:
        params = "ptr swiftasync %ctx, " + params
    attrs = ' "frame-pointer"="all"' if case.fp else ""
    out = ["define %s void @test_%s(%s)%s {" % (cc, case.name, params, attrs),
           "entry:"]
    if case.dynamic:
        out += ["  %dyn = alloca i8, i64 %p1",
                "  %al = alloca [4 x i64], align 64",
                "  call void @use(ptr %dyn)", "  call void @use(ptr %al)"]
    if case.xmm:
        out.append("  %d = load volatile double, ptr @dval")
    if case.async_:
        out += ["  %ca = call ptr @llvm.swift.async.context.addr()",
                "  call void @use(ptr %ca)"]
    for i in sorted(case.hoist):
        out.append("  %%hold%d = load volatile i64, ptr @hold%d" % (i, i))
    if case.noop:
        out.append("  call void @noop()")
    for s_, (kind_, _m) in enumerate(case.paths):
        if kind_ == "mem":
            out.append("  %%slot%d = alloca ptr" % s_)
    labels = " ".join("i64 %d, label %%s%d" % (s, s)
                      for s, _ in enumerate(case.paths)
                      if case.paths[s][0] != "ret")
    out.append("  switch i64 %%sel, label %%ret [ %s ]" % labels)
    for s, (kind, m) in enumerate(case.paths):
        if kind == "ret":
            continue
        out.append("s%d:" % s)
        callee = None
        if kind == "int" or kind == "mem" or kind == "async":
            defs, args = int_args(case, m, "v%d_" % s)
            out += defs
            if kind == "async":
                args = ["ptr swiftasync %ctx"] + args
                callee = "@target_a%d" % m
            else:
                callee = "@target_i%d" % m
        elif kind == "dbl":
            ni, nd = m
            defs, args = [], []
            for i in range(1, ni + 1):
                if i <= case.k:
                    args.append("i64 %%p%d" % i)
                else:
                    defs.append("  %%v%d_%d = add i64 %%p1, %d" % (s, i, i - 1))
                    args.append("i64 %%v%d_%d" % (s, i))
            out += defs
            args += ["double %d" for _ in range(nd)]
            callee = "@target_d%d_%d" % (ni, nd)
        elif kind == "huge":
            args = ["[%d x i64] zeroinitializer" % HUGE]
            callee = "@target_huge"
        if kind == "mem":
            out += ["  store ptr %s, ptr %%slot%d" % (callee, s),
                    "  call void @use(ptr %%slot%d)" % s,
                    "  %%fp%d = load ptr, ptr %%slot%d" % (s, s)]
            callee = "%%fp%d" % s
        out.append("  musttail call %s void %s(%s)" % (cc, callee,
                                                       ", ".join(args)))
        out.append("  ret void")
    out += ["ret:", "  ret void", "}"]
    return "\n".join(out)


def runner(case):
    skip = ("r13", "r14") if case.async_ else ()
    asm, clob = sentinel_asm(skip)
    if case.large is not None:
        args = "i64 %%sel64, [%d x i64] zeroinitializer" % case.large
    else:
        args = ", ".join(["i64 %sel64"] +
                         ["i64 %d" % (1000 + i) for i in range(1, case.k + 1)])
    if case.async_:
        args = "ptr swiftasync inttoptr (i64 4660 to ptr), " + args
    return "\n".join([
        "define i32 @run_%s(i32 %%selector) {" % case.name,
        "  store volatile i32 0, ptr @args_ok",
        '  call void asm sideeffect "%s", "%s"()' % (asm, clob),
        "  call void @RtlCaptureContext(ptr @unwind_expected_context)",
        "  %sel64 = zext i32 %selector to i64",
        "  call %s void @test_%s(%s)" % (case.cc, case.name, args),
        "  call void @RtlCaptureContext(ptr @unwind_actual_context)",
        "  %ok = load volatile i32, ptr @args_ok",
        "  ret i32 %ok", "}"])


def emit(filename, title, cases, extra_doc=""):
    ints, dbls, asyncs = set(), set(), set()
    huge = False
    for c in cases:
        for kind, m in c.paths:
            if kind in ("int", "mem"):
                ints.add(m)
            elif kind == "dbl":
                dbls.add(m)
            elif kind == "async":
                asyncs.add(m)
            elif kind == "huge":
                huge = True
    uses_async = any(c.async_ for c in cases)
    o = []
    o.append("; REQUIRES: system-windows")
    o.append("; REQUIRES: target={{x86_64.*-windows-msvc}}")
    for opt in ("", "-O0 "):
        tag = "O0" if opt else "O2"
        o.append("; RUN: llc -mtriple=x86_64-pc-windows-msvc %s-filetype=obj %%s -o %%t.%s.obj" % (opt, tag))
        o.append("; RUN: %%python %%S/Inputs/build-win64-unwind-harness.py %%S/Inputs/win64-unwind-harness.c %%t.%s.obj %%t.%s.exe" % (tag, tag))
        o.append("; RUN: %%t.%s.exe" % tag)
    o.append("")
    o.append("; NOTE: Generated by Inputs/gen-win64-unwind-tests.py. Do not edit by hand.")
    o.append("")
    o.append("; " + title)
    o.append(";")
    o.append("; Run under Windows. The harness (Inputs/win64-unwind-harness.c) runs each")
    o.append("; case to completion, then single-steps it: at each instruction of the")
    o.append("; function under test, and of the function it tail calls, the unwinder must")
    o.append("; recover the caller and its nonvolatile registers, and the system exception")
    o.append("; dispatcher must be able to unwind to a handler in the harness.")
    if extra_doc:
        o.append(";")
        for l in extra_doc.split("\n"):
            o.append("; " + l)
    o.append("")
    o.append("declare void @RtlCaptureContext(ptr)")
    if uses_async:
        o.append("declare ptr @llvm.swift.async.context.addr()")
    o.append("")
    o.append("@unwind_expected_context = global [1232 x i8] zeroinitializer, align 16")
    o.append("@unwind_actual_context = global [1232 x i8] zeroinitializer, align 16")
    o.append("@args_ok = global i32 0")
    for i in sorted({i for c in cases for i in c.hoist}):
        o.append("@hold%d = global i64 %d" % (i, 1000 + i))
    if any(c.xmm for c in cases):
        o.append("@dval = global double 1.001000e+03")
    o.append("")
    o.append("define void @noop() noinline {\n  ret void\n}")
    o.append("define void @use(ptr %p) noinline {\n  ret void\n}")
    o.append("")
    checked = []
    for m in sorted(ints):
        o += [int_target(m), ""]
        checked.append("target_i%d" % m)
    for m in sorted(dbls):
        o += [dbl_target(*m), ""]
        checked.append("target_d%d_%d" % m)
    for m in sorted(asyncs):
        o += [async_target(m), ""]
        checked.append("target_a%d" % m)
    if huge:
        o += [huge_target(), ""]
        checked.append("target_huge")
    allfns = ["noop", "use"] + list(checked)
    for c in cases:
        o += [test_function(c), "", runner(c), ""]
        checked.append("test_%s" % c.name)
        allfns += ["test_%s" % c.name, "run_%s" % c.name]
    # Defined last, so that the end of the last function is known. Functions
    # without unwind data (frameless leaves) are found by address range.
    o += ["define void @unwind_end_marker() noinline {\n  ret void\n}", ""]
    allfns.append("unwind_end_marker")
    o.append("%TestCase = type { ptr, ptr, i64, i64, i64 }")
    entries = []
    for i, c in enumerate(cases):
        nm = c.name
        o.append('@.name.%d = private unnamed_addr constant [%d x i8] c"%s\\00"' % (i, len(nm) + 1, nm))
        mask = sum(1 << s for s, (k, m) in enumerate(c.paths) if k != "ret")
        flags = 1 if c.async_ else 0
        entries.append("  %%TestCase { ptr @.name.%d, ptr @run_%s, i64 %d, i64 %d, i64 %d }" % (i, nm, len(c.paths), mask, flags))
    entries.append("  %TestCase zeroinitializer")
    o.append("@unwind_test_cases = constant [%d x %%TestCase] [\n%s\n]" % (len(entries), ",\n".join(entries)))
    cl = ["  ptr @%s" % n for n in checked] + ["  ptr null"]
    o.append("@unwind_checked_functions = constant [%d x ptr] [\n%s\n]" % (len(cl), ",\n".join(cl)))
    al = ["  ptr @%s" % n for n in allfns] + ["  ptr null"]
    o.append("@unwind_all_functions = constant [%d x ptr] [\n%s\n]" % (len(al), ",\n".join(al)))
    with open(os.path.join(OUT, filename), "w") as f:
        f.write("\n".join(o) + "\n")


GROW = [("int", 12), ("int", 10), ("int", 7), ("int", 4), ("ret", 0)]
SHRINK = [("int", 7), ("int", 11), ("int", 12), ("ret", 0)]

emit("win64-tailcc-unwind-frames.ll",
     "Unwinding through tail calls that grow and shrink the stack argument area.",
     [Case("grow_nofp", GROW),
      Case("grow_fp", GROW, fp=True),
      Case("grow_csr", GROW, noop=True),
      Case("grow_csr_fp", GROW, noop=True, fp=True),
      Case("grow_csr_src", GROW, noop=True, consts=True),
      Case("grow_csr_src_fp", GROW, noop=True, consts=True, fp=True),
      Case("grow_dyn", GROW, noop=True, fp=True, dynamic=True),
      Case("shrink_nofp", SHRINK, k=11),
      Case("shrink_csr", SHRINK, k=11, noop=True),
      Case("shrink_fp", SHRINK, k=11, noop=True, fp=True),
      Case("mem_target", [("mem", 12), ("mem", 10), ("mem", 7), ("ret", 0)], noop=True)],
     "Selectors go from the tail call that needs the most stack down to one with\nthe same size as the caller's own arguments (growing cases), or from the\nlargest shrink (shrinking cases); the last selector is a return. The csr\ncases keep values in callee-saved registers across a call, including the one\nstored over the caller's return address.")

XMM = [("dbl", (4, 8)), ("dbl", (4, 6)), ("dbl", (4, 3)), ("int", 4), ("ret", 0)]
emit("win64-tailcc-unwind-xmm.ll",
     "Unwinding when the argument stored over the return address is in a\n; callee-saved XMM register.",
     [Case("xmm_nofp", XMM, noop=True, xmm=True),
      Case("xmm_fp", XMM, noop=True, xmm=True, fp=True)])

ASYNC = [("async", 12), ("async", 10), ("async", 7), ("async", 4), ("ret", 0)]
emit("win64-tailcc-unwind-swiftasync.ll",
     "Unwinding through swifttailcc functions with a swiftasync parameter.",
     [Case("async_nofp", ASYNC, cc="swifttailcc", async_=True, noop=True),
      Case("async_fp", ASYNC, cc="swifttailcc", async_=True, noop=True, fp=True)])

emit("win64-tailcc-unwind-large.ll",
     "Unwinding when the reserve exceeds a page, and when a function pops more\n; than 65535 bytes on return.",
     [Case("pop_nofp", [("ret", 0), ("int", 7)], large=BIG),
      Case("pop_csr", [("ret", 0), ("int", 7)], large=BIG, noop=True),
      Case("pop_fp", [("ret", 0), ("int", 7)], large=BIG, noop=True, fp=True),
      Case("reserve_nofp", [("huge", 0), ("ret", 0)]),
      Case("reserve_fp", [("huge", 0), ("ret", 0)], noop=True, fp=True)])
