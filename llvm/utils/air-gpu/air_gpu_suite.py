#!/usr/bin/env python3
"""Hardware test suite for the AIR target.

Generates compute kernels that exercise AIR functionality, compiles them with
the AIR backend (llc -mtriple=air -filetype=obj -air-output=air), runs them on
an Apple GPU through the AIR service (/air, see air_gpu_run.py) and compares
every result with a reference computed here in Python.

Every kernel is generated from text in this file and compiled by our own
backend. The service refuses Metal source, so no output of Apple's Metal
compiler can enter a test.

For each case the suite reports one of:
  PASS      the device accepted the kernel and the results match
  MISMATCH  the device ran the kernel and some result is wrong
  REJECTED  the device refused the library (the case is then retried alone,
            so a rejection is attributed to the one kernel that causes it)
  COMPILE   llc failed
  ABSENT    the device rejected a probe for a function AIR does not have, as
            expected (these record what the catalogue must not emit)

  air_gpu_suite.py --llc build/bin/llc [--group math-f32 ...] [--json out.json]
"""

import argparse
import base64
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.environ.get('AIR_CLIENT', '/air/client'))
from client import submit  # noqa: E402

N = 64  # threads per case: two SIMD-groups

TYPES = {
    'f32': ('float', 'f', 4), 'f16': ('half', 'e', 2),
    'i32': ('i32', 'i', 4), 'u32': ('i32', 'I', 4),
    'i16': ('i16', 'h', 2), 'u16': ('i16', 'H', 2),
    'i8': ('i8', 'b', 1), 'u8': ('i8', 'B', 1),
    'i64': ('i64', 'q', 8), 'u64': ('i64', 'Q', 8),
    'i1': ('i1', 'B', 1),
}


def f32(x):
    """Round a Python float to float32."""
    try:
        return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f16(x):
    try:
        return struct.unpack('<e', struct.pack('<e', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def bits32(x):
    return struct.unpack('<i', struct.pack('<f', x))[0]


def ulp_distance(a, b):
    """Distance in float32 ULPs; 0 for two NaNs."""
    if math.isnan(a) or math.isnan(b):
        return 0 if math.isnan(a) and math.isnan(b) else 1 << 31
    ia, ib = bits32(a), bits32(b)
    if ia < 0:
        ia = -(1 << 31) - ia
    if ib < 0:
        ib = -(1 << 31) - ib
    return abs(ia - ib)


class Case:
    """One kernel: out[tid] = body(in0[tid], in1[tid], ...).

    body is LLVM IR text using %a0, %a1, ... (the loaded inputs) and must
    define %r (the result). decls are the declarations it needs.
    ref(i, inputs) returns the expected value for thread i. inputs are lists of
    N values. ulps (floats) is the tolerance, None for exact comparison.
    """

    def __init__(self, name, ins, out, body, decls, ref, gen=None, ulps=None,
                 source='', tgsize=N, full=None, abstol=None, absent=False):
        self.name = name
        self.ins = ins
        self.out = out
        self.body = body
        self.decls = decls
        self.ref = ref
        self.gen = gen or [None] * len(ins)
        self.ulps = ulps
        self.source = source
        self.tgsize = tgsize
        self.full = full  # a complete kernel body instead of the template
        self.abstol = abstol  # floats: also accept |got - want| <= abstol
        self.absent = absent  # a probe for a function AIR does not have


def kernel_ir(c):
    if c.full is not None:
        return c.full
    args = ', '.join(['ptr addrspace(1) %%in%d' % i for i in range(len(c.ins))] +
                     ['ptr addrspace(1) %out'])
    lines = ['define void @%s(%s) {' % (c.name, args), 'entry:',
             '  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()',
             '  %tid = extractvalue { i32, i32, i32 } %t, 0']
    for i, ty in enumerate(c.ins):
        irty = TYPES[ty][0]
        lines += ['  %%p%d = getelementptr %s, ptr addrspace(1) %%in%d, i32 %%tid' % (i, irty, i),
                  '  %%a%d = load %s, ptr addrspace(1) %%p%d' % (i, irty, i)]
    lines += ['  ' + l for l in c.body.strip().splitlines()]
    oty = TYPES[c.out][0]
    lines += ['  %%po = getelementptr %s, ptr addrspace(1) %%out, i32 %%tid' % oty,
              '  store %s %%r, ptr addrspace(1) %%po' % oty, '  ret void', '}']
    return '\n'.join(lines)


MODULE_SUFFIX = ''


def module_ir(cases):
    decls = {'declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()'}
    for c in cases:
        decls.update(c.decls)
    return ('\n'.join(sorted(decls)) + '\n\n' + '\n\n'.join(kernel_ir(c) for c in cases) + '\n' +
            MODULE_SUFFIX)


def pack(ty, values):
    code = TYPES[ty][1]
    if ty == 'f32':
        values = [f32(v) for v in values]
    return struct.pack('<%d%s' % (len(values), code), *values)


def unpack(ty, data, n):
    code, size = TYPES[ty][1], TYPES[ty][2]
    return list(struct.unpack('<%d%s' % (n, code), data[:n * size]))


def default_gen(ty, rng):
    if ty == 'f32':
        return rng.uniform(-8, 8)
    if ty == 'f16':
        return f16(rng.uniform(-8, 8))
    size = TYPES[ty][2] * 8
    if ty.startswith('u'):
        return rng.randrange(0, 1 << size)
    if ty == 'i1':
        return rng.randrange(0, 2)
    return rng.randrange(-(1 << (size - 1)), 1 << (size - 1))


def compile_air(llc, ir, tmp, tag):
    ll = os.path.join(tmp, tag + '.ll')
    air = os.path.join(tmp, tag + '.air')
    open(ll, 'w').write(ir)
    r = subprocess.run([llc, '-mtriple=air', '-filetype=obj', '-air-output=air', ll, '-o', air],
                       capture_output=True, text=True)
    if r.returncode:
        return None, r.stderr.strip()[-400:]
    return open(air, 'rb').read(), None


TRANSIENT = ('XPC_ERROR_CONNECTION_INTERRUPTED', 'did not reply')


def call(job, queue, tries=4):
    """submit(), retrying the service's transient failures."""
    for attempt in range(tries):
        try:
            reply = submit(job, queue, timeout=120)
        except TimeoutError as e:
            reply = {'ok': False, 'error': str(e)}
        if reply.get('ok') or not any(t in reply.get('error', '') for t in TRANSIENT):
            return reply
    return reply


def run_case(c, air, inputs, queue):
    buffers = [{'size': len(d), 'data': base64.b64encode(d).decode()} for d in inputs]
    buffers.append({'size': N * TYPES[c.out][2]})
    job = {'op': 'execute', 'air': base64.b64encode(air).decode(), 'buffers': buffers,
           'commands': [{'kernel': c.name,
                         'bindings': [{'index': i, 'buffer': i} for i in range(len(buffers))],
                         'grid': [N, 1, 1], 'group': [c.tgsize, 1, 1]}],
           'readback': [len(buffers) - 1]}
    return call(job, queue)


def check(c, values, got):
    want = [c.ref(i, values) for i in range(N)]
    worst, bad = 0, []
    for i, (g, w) in enumerate(zip(got, want)):
        if c.ulps is None:
            ok = g == w
            if not ok:
                bad.append((i, g, w))
        else:
            d = ulp_distance(g, f32(w) if c.out == 'f32' else w)
            worst = max(worst, d)
            near = c.abstol is not None and abs(g - w) <= c.abstol
            if d > c.ulps and not near:
                bad.append((i, g, w))
    return bad, worst


def run_group(cases, llc, queue, tmp, log):
    rng = random.Random(1234)
    data = {}
    for c in cases:
        values = []
        for ty, gen in zip(c.ins, c.gen):
            vals = [(gen(i, rng) if gen else default_gen(ty, rng)) for i in range(N)]
            # The device sees the values as stored: the reference must too.
            if ty == 'f32':
                vals = [f32(x) for x in vals]
            elif ty == 'f16':
                vals = [f16(x) for x in vals]
            values.append(vals)
        data[c.name] = values

    results = {}
    air, err = compile_air(llc, module_ir(cases), tmp, 'group')
    rejected_together = False
    if air is None:
        log('  group failed to compile, compiling cases alone')
    else:
        reply = call({'op': 'inspect', 'air': base64.b64encode(air).decode()}, queue)
        rejected_together = not reply.get('ok')

    for c in cases:
        mine = air
        if air is None or rejected_together:
            mine, err = compile_air(llc, module_ir([c]), tmp, c.name)
            if mine is None:
                results[c.name] = {'status': 'COMPILE', 'detail': err}
                continue
        values = data[c.name]
        reply = run_case(c, mine, [pack(t, v) for t, v in zip(c.ins, values)], queue)
        if not reply.get('ok'):
            results[c.name] = {'status': 'ABSENT' if c.absent else 'REJECTED',
                               'detail': reply.get('error', '')}
            continue
        if c.absent:
            results[c.name] = {'status': 'MISMATCH', 'detail': 'expected the device to reject it'}
            continue
        raw = base64.b64decode(reply['result']['buffers'][0]['data'])
        got = unpack(c.out, raw, N)
        bad, worst = check(c, values, got)
        r = {'status': 'MISMATCH' if bad else 'PASS'}
        if c.ulps is not None:
            r['max_ulps'] = worst
        if bad:
            i, g, w = bad[0]
            r['detail'] = '%d/%d wrong; thread %d: got %r want %r (inputs %r)' % (
                len(bad), N, i, g, w, [v[i] for v in values])
        results[c.name] = r
    return results


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def unary_f32(name, fn, ref, ulps, gen=None, flags='', source=''):
    tag = name.replace('.', '_')
    return Case(tag, ['f32'], 'f32',
                '%%r = call %s float @%s(float %%a0)' % (flags, fn),
                {'declare float @%s(float)' % fn},
                lambda i, v, ref=ref: ref(v[0][i]), [gen], ulps, source)


def binary_f32(name, fn, ref, ulps, gen=None, gen2=None, source=''):
    tag = name.replace('.', '_')
    return Case(tag, ['f32', 'f32'], 'f32',
                '%%r = call float @%s(float %%a0, float %%a1)' % fn,
                {'declare float @%s(float, float)' % fn},
                lambda i, v, ref=ref: ref(v[0][i], v[1][i]), [gen, gen2], ulps, source)


def pos(i, rng):
    return rng.uniform(0.01, 16)


def unit(i, rng):
    return rng.uniform(-0.99, 0.99)


def ge1(i, rng):
    return rng.uniform(1.0, 16)


def small(i, rng):
    return rng.uniform(-4, 4)


def safe(fn):
    def wrapped(*a):
        try:
            return fn(*a)
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.nan
    return wrapped


# (name, reference, input domain, precise ULPs) for float functions that exist
# both as an LLVM intrinsic and as an AIR function.
FLOAT_UNARY = [
    ('fabs', abs, None, 0), ('floor', math.floor, None, 0), ('ceil', math.ceil, None, 0),
    ('trunc', math.trunc, None, 0), ('rint', lambda x: float(round(x)), None, 0),
    ('round', lambda x: math.copysign(math.floor(abs(x) + 0.5), x), None, 0),
    ('sqrt', math.sqrt, pos, 0), ('exp', math.exp, small, 4), ('exp2', lambda x: 2.0 ** x, small, 4),
    ('exp10', lambda x: 10.0 ** x, small, 4), ('log', math.log, pos, 4), ('log2', math.log2, pos, 4),
    ('log10', math.log10, pos, 4), ('sin', math.sin, None, 4), ('cos', math.cos, None, 4),
    ('tan', math.tan, unit, 6), ('asin', math.asin, unit, 4), ('acos', math.acos, unit, 4),
    ('atan', math.atan, None, 5), ('sinh', math.sinh, small, 4), ('cosh', math.cosh, small, 4),
    ('tanh', math.tanh, small, 5), ('asinh', math.asinh, None, 4), ('acosh', math.acosh, ge1, 4),
    ('atanh', math.atanh, unit, 5),
]

AIR_ONLY_UNARY = [
    ('rsqrt', lambda x: 1 / math.sqrt(x), pos, 2), ('fract', lambda x: x - math.floor(x), None, 0),
    ('sinpi', lambda x: math.sin(math.pi * x), small, 4), ('cospi', lambda x: math.cos(math.pi * x), small, 4),
    ('tanpi', lambda x: math.tan(math.pi * x), lambda i, r: r.uniform(-0.2, 0.2), 6),
]

# AIR functions with no LLVM intrinsic (a function called llvm.asinh is just an
# undefined external, which no front end emits).
NOT_LLVM_INTRINSICS = {'asinh', 'acosh', 'atanh'}

FLOAT_BINARY = [
    ('pow', math.pow, pos, None, 16), ('atan2', math.atan2, None, None, 6),
    ('fmin', min, None, None, 0), ('fmax', max, None, None, 0),
    ('copysign', math.copysign, None, None, 0), ('fmod', math.fmod, None, pos, 0),
]


def group_llvm_math_f32():
    """Generic LLVM math intrinsics, as a front end emits them."""
    cases = []
    for name, ref, gen, ulps in FLOAT_UNARY:
        if name in NOT_LLVM_INTRINSICS:
            continue
        cases.append(unary_f32('llvm.' + name, 'llvm.%s.f32' % name, safe(ref), ulps, gen,
                               source='LLVM intrinsic'))
    for name, llvm in [('fmin', 'minnum'), ('fmax', 'maxnum'), ('pow', 'pow'),
                       ('atan2', 'atan2'), ('copysign', 'copysign')]:
        ref = {'fmin': min, 'fmax': max, 'pow': math.pow, 'atan2': math.atan2,
               'copysign': math.copysign}[name]
        gen = pos if name == 'pow' else None
        cases.append(binary_f32('llvm.' + llvm, 'llvm.%s.f32' % llvm, safe(ref),
                                16 if name == 'pow' else (6 if name == 'atan2' else 0), gen))
    cases.append(Case('llvm_fma', ['f32', 'f32', 'f32'], 'f32',
                      '%r = call float @llvm.fma.f32(float %a0, float %a1, float %a2)',
                      {'declare float @llvm.fma.f32(float, float, float)'},
                      lambda i, v: v[0][i] * v[1][i] + v[2][i], ulps=1))
    return cases


def group_air_math_f32():
    """The same functions called as AIR functions directly (Metal.jl, dxmt, floor)."""
    cases = []
    for name, ref, gen, ulps in FLOAT_UNARY + AIR_ONLY_UNARY:
        cases.append(unary_f32('air.' + name, 'air.%s.f32' % name, safe(ref), ulps, gen,
                               source='Metal.jl math.jl, floor MetalFinal'))
    for name, ref, gen, ulps in FLOAT_UNARY + AIR_ONLY_UNARY:
        if name in ('fabs', 'floor', 'ceil', 'trunc', 'rint', 'round', 'fract', 'sqrt'):
            continue
        c = unary_f32('air.fast_' + name, 'air.fast_%s.f32' % name, safe(ref),
                      max(64, ulps * 16), gen, source='Metal.jl fast variants')
        c.abstol = 2.0 ** -12  # fast functions are specified by absolute error
        cases.append(c)
    for name, ref, gen, gen2, ulps in FLOAT_BINARY:
        c = binary_f32('air.' + name, 'air.%s.f32' % name, safe(ref), ulps, gen, gen2)
        # There is no air.copysign: llvm.copysign stays an LLVM intrinsic.
        c.absent = name == 'copysign'
        cases.append(c)
    cases.append(Case('air_fma', ['f32', 'f32', 'f32'], 'f32',
                      '%r = call float @air.fma.f32(float %a0, float %a1, float %a2)',
                      {'declare float @air.fma.f32(float, float, float)'},
                      lambda i, v: v[0][i] * v[1][i] + v[2][i], ulps=1, source='dxmt CreateFMA'))
    return cases


def group_llvm_math_afn():
    """afn calls, which must reach the relaxed air.fast_* functions."""
    cases = []
    for name, ref, gen, ulps in FLOAT_UNARY:
        if name in NOT_LLVM_INTRINSICS or name in ('fabs', 'floor', 'ceil', 'trunc', 'rint', 'round'):
            continue
        c = unary_f32('llvm.afn.' + name, 'llvm.%s.f32' % name, safe(ref), max(64, ulps * 16), gen,
                      flags='afn', source='afn')
        c.abstol = 2.0 ** -12
        cases.append(c)
    return cases


def int_case(name, intr, ty, nargs, ref, extra='', gen=None):
    irty = TYPES[ty][0]
    args = ', '.join('%s %%a%d' % (irty, i) for i in range(nargs)) + extra
    params = ', '.join([irty] * nargs) + (', i1' if extra else '')
    return Case(name, [ty] * nargs, ty, '%%r = call %s @%s(%s)' % (irty, intr, args),
                {'declare %s @%s(%s)' % (irty, intr, params)}, ref,
                [gen] * nargs if gen else None)


def s32(x):
    return ((x + (1 << 31)) % (1 << 32)) - (1 << 31)


def u32(x):
    return x % (1 << 32)


def group_llvm_int():
    """Integer intrinsics: the signedness infix and operand trimming."""
    def u(i, v, k):
        return u32(v[k][i])
    small_pos = lambda i, r: r.randrange(0, 1 << r.randrange(1, 32))
    return [
        int_case('llvm_smin_i32', 'llvm.smin.i32', 'i32', 2, lambda i, v: min(v[0][i], v[1][i])),
        int_case('llvm_smax_i32', 'llvm.smax.i32', 'i32', 2, lambda i, v: max(v[0][i], v[1][i])),
        int_case('llvm_umin_i32', 'llvm.umin.i32', 'u32', 2, lambda i, v: min(v[0][i], v[1][i])),
        int_case('llvm_umax_i32', 'llvm.umax.i32', 'u32', 2, lambda i, v: max(v[0][i], v[1][i])),
        int_case('llvm_abs_i32', 'llvm.abs.i32', 'i32', 1, lambda i, v: s32(abs(v[0][i])), ', i1 false'),
        int_case('llvm_ctpop_i32', 'llvm.ctpop.i32', 'u32', 1, lambda i, v: bin(v[0][i]).count('1')),
        int_case('llvm_ctlz_i32', 'llvm.ctlz.i32', 'u32', 1, lambda i, v: 32 - v[0][i].bit_length(),
                 ', i1 false', small_pos),
        int_case('llvm_cttz_i32', 'llvm.cttz.i32', 'u32', 1,
                 lambda i, v: 32 if v[0][i] == 0 else (v[0][i] & -v[0][i]).bit_length() - 1,
                 ', i1 false', small_pos),
        int_case('llvm_bitreverse_i32', 'llvm.bitreverse.i32', 'u32', 1,
                 lambda i, v: int(format(v[0][i], '032b')[::-1], 2)),
    ]


GROUPS = {
    'llvm-math-f32': group_llvm_math_f32,
    'air-math-f32': group_air_math_f32,
    'llvm-math-afn': group_llvm_math_afn,
    'llvm-int': group_llvm_int,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--llc', required=True)
    ap.add_argument('--group', action='append', choices=sorted(GROUPS))
    ap.add_argument('--queue', default=os.environ.get('AIR_QUEUE', '/air'))
    ap.add_argument('--json', help='write every result here')
    ap.add_argument('--module-suffix', default='',
                    help='IR text appended to every module, e.g. named metadata')
    ap.add_argument('--keep', help='keep the generated .ll/.air files in this directory')
    args = ap.parse_args()

    global MODULE_SUFFIX
    MODULE_SUFFIX = args.module_suffix.replace('\\n', '\n')
    all_results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = args.keep or tmp
        os.makedirs(tmp, exist_ok=True)
        for g in args.group or sorted(GROUPS):
            print('== %s' % g)
            res = run_group(GROUPS[g](), args.llc, args.queue, tmp, print)
            for name, r in res.items():
                extra = (' max_ulps=%d' % r['max_ulps']) if 'max_ulps' in r else ''
                print('%-9s %-28s%s %s' % (r['status'], name, extra, r.get('detail', '')))
            all_results[g] = res
    counts = {}
    for res in all_results.values():
        for r in res.values():
            counts[r['status']] = counts.get(r['status'], 0) + 1
    print('summary: ' + ', '.join('%s %d' % kv for kv in sorted(counts.items())))
    if args.json:
        json.dump(all_results, open(args.json, 'w'), indent=1)
    return 1 if set(counts) - {'PASS', 'ABSENT'} else 0


if __name__ == '__main__':
    sys.exit(main())
