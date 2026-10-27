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
  HOST-LIMIT the device rejected a case that needs a newer OS than it runs
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
                 source='', tgsize=N, full=None, abstol=None, absent=False, out_init=None,
                 scalars=None, host_limit=None):
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
        self.out_init = out_init  # initial contents of the output buffer
        # Packed scalar arguments: the kernel is (out, scalars...), and the
        # backend passes the scalars in a buffer bound after out.
        self.scalars = scalars
        # Needs something this device's OS lacks (a reason): a rejection is
        # then HOST-LIMIT, not a failure.
        self.host_limit = host_limit


def kernel_ir(c):
    if c.full is not None:
        return c.full
    args = ', '.join(['ptr addrspace(1) %%in%d' % i for i in range(len(c.ins))] +
                     ['ptr addrspace(1) %out'])
    lines = ['define void @%s(%s) {' % (c.name, args), 'entry:',
             '  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()',
             '  %tid = extractvalue { i32, i32, i32 } %t, 0',
             '  %lane = and i32 %tid, 31', '  %lane16 = trunc i32 %lane to i16']
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
    if c.scalars is not None:
        out = {'size': N * TYPES[c.out][2]}
        packed = b''.join(struct.pack('<' + TYPES[t][1], v) for t, v in c.scalars)
        buffers = [out, {'size': len(packed), 'data': base64.b64encode(packed).decode()}]
        job = {'op': 'execute', 'air': base64.b64encode(air).decode(), 'buffers': buffers,
               'commands': [{'kernel': c.name,
                             'bindings': [{'index': i, 'buffer': i} for i in range(2)],
                             'grid': [N, 1, 1], 'group': [c.tgsize, 1, 1]}],
               'readback': [0]}
        return call(job, queue)
    buffers = [{'size': len(d), 'data': base64.b64encode(d).decode()} for d in inputs]
    out = {'size': N * TYPES[c.out][2]}
    if c.out_init is not None:
        out['data'] = base64.b64encode(pack(c.out, c.out_init)).decode()
    buffers.append(out)
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
        if c.ulps is None and c.abstol is not None:
            if not (abs(g - w) <= c.abstol or (g != g and w != w)):
                bad.append((i, g, w))
        elif c.ulps is None:
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
            results[c.name] = {'status': 'ABSENT' if c.absent else
                               'HOST-LIMIT' if c.host_limit else 'REJECTED',
                               'detail': (c.host_limit + ': ' if c.host_limit else '') +
                               reply.get('error', '')}
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


def lanes(i, values):
    """The inputs of the threads of thread i's SIMD-group."""
    base = i // 32 * 32
    return values[base:base + 32]


def simd_case(name, ty, fn_suffix, call, ref, extra_decl_args='', ulps=None, abstol=None, gen=None):
    irty = TYPES[ty][0]
    decl = 'declare %s @%s.%s(%s)' % (irty, call[0], fn_suffix, ', '.join([irty] + list(call[2])))
    body = '%%r = call %s @%s.%s(%s %%a0%s)' % (irty, call[0], fn_suffix, irty, call[1])
    c = Case(name, [ty], ty, body, {decl}, ref, [gen], ulps, 'floor metal_group.hpp, Metal.jl simd.jl')
    c.abstol = abstol
    return c


def group_simd():
    """SIMD-group and quad-group operations (two SIMD-groups of 32)."""
    cases = []
    for ty, sfx in (('i32', 's.i32'), ('u32', 'u.i32'), ('f32', 'f32')):
        w = (lambda x: x) if ty == 'f32' else (s32 if ty == 'i32' else u32)
        tag = sfx.replace('.', '_')
        cases += [
            simd_case('simd_shuffle_' + tag, ty, sfx,
                      ('air.simd_shuffle', ', i16 %next', ['i16']),
                      lambda i, v: lanes(i, v[0])[(i % 32 + 1) % 32]),
            simd_case('simd_shuffle_xor_' + tag, ty, sfx,
                      ('air.simd_shuffle_xor', ', i16 5', ['i16']),
                      lambda i, v: lanes(i, v[0])[(i % 32) ^ 5]),
            simd_case('simd_shuffle_up_' + tag, ty, sfx,
                      ('air.simd_shuffle_up', ', i16 3', ['i16']),
                      lambda i, v: v[0][i] if i % 32 < 3 else v[0][i - 3]),
            simd_case('simd_shuffle_down_' + tag, ty, sfx,
                      ('air.simd_shuffle_down', ', i16 3', ['i16']),
                      lambda i, v: v[0][i] if i % 32 >= 29 else v[0][i + 3]),
            simd_case('quad_shuffle_xor_' + tag, ty, sfx,
                      ('air.quad_shuffle_xor', ', i16 1', ['i16']),
                      lambda i, v: v[0][i ^ 1]),
            simd_case('simd_sum_' + tag, ty, sfx, ('air.simd_sum', '', []),
                      lambda i, v, w=w: w(sum(lanes(i, v[0]))),
                      ulps=64 if ty == 'f32' else None, abstol=1e-4 if ty == 'f32' else None),
            simd_case('simd_min_' + tag, ty, sfx, ('air.simd_min', '', []),
                      lambda i, v: min(lanes(i, v[0])), ulps=0 if ty == 'f32' else None),
            simd_case('simd_max_' + tag, ty, sfx, ('air.simd_max', '', []),
                      lambda i, v: max(lanes(i, v[0])), ulps=0 if ty == 'f32' else None),
            simd_case('simd_prefix_inclusive_sum_' + tag, ty, sfx,
                      ('air.simd_prefix_inclusive_sum', '', []),
                      lambda i, v, w=w: w(sum(lanes(i, v[0])[:i % 32 + 1])),
                      ulps=64 if ty == 'f32' else None, abstol=1e-4 if ty == 'f32' else None),
            simd_case('simd_prefix_exclusive_sum_' + tag, ty, sfx,
                      ('air.simd_prefix_exclusive_sum', '', []),
                      lambda i, v, w=w: w(sum(lanes(i, v[0])[:i % 32])),
                      ulps=64 if ty == 'f32' else None, abstol=1e-4 if ty == 'f32' else None),
        ]
        if ty != 'f32':
            import functools, operator
            for op, fn in (('and', operator.and_), ('or', operator.or_), ('xor', operator.xor)):
                cases.append(simd_case('simd_%s_%s' % (op, tag), ty, sfx, ('air.simd_' + op, '', []),
                                       lambda i, v, fn=fn, w=w: w(functools.reduce(fn, lanes(i, v[0])))))
    for c in cases:
        if c.ulps is None and c.ins == ['f32']:
            c.ulps = 0
        if '%next' in c.body:
            c.body = '%nl = add i32 %lane, 1\n%nm = and i32 %nl, 31\n%next = trunc i32 %nm to i16\n' + c.body
    # Ballot: bit k of the low word is thread k's predicate (odd input).
    cases.append(Case('simd_ballot', ['u32'], 'u32',
                      '%odd = trunc i32 %a0 to i1\n%m = call i64 @air.simd_ballot.i64(i1 %odd)\n'
                      '%r = trunc i64 %m to i32',
                      {'declare i64 @air.simd_ballot.i64(i1)'},
                      lambda i, v: sum((x & 1) << k for k, x in enumerate(lanes(i, v[0]))),
                      source='Metal.jl simd_ballot'))
    cases.append(Case('simd_vote_any', ['u32'], 'u32',
                      '%big = icmp ugt i32 %a0, 4000000000\n%m = call i64 @air.simd_ballot.i64(i1 %big)\n'
                      '%b = call i1 @air.simd_vote_any.i64(i64 %m)\n%r = zext i1 %b to i32',
                      {'declare i64 @air.simd_ballot.i64(i1)', 'declare i1 @air.simd_vote_any.i64(i64)'},
                      lambda i, v: int(any(x > 4000000000 for x in lanes(i, v[0]))),
                      source='Metal.jl simd_vote_any'))
    return cases


SLOTS = 8


def fold_slots(values, fn, init, wrap):
    """out[j] = fn folded over the inputs of threads i with i % SLOTS == j."""
    out = [init] * N
    for i, x in enumerate(values):
        out[i % SLOTS] = wrap(fn(out[i % SLOTS], x))
    return out


def atomic_case(name, op, ty, fn, init=0, space='device', ordering='monotonic'):
    irty = TYPES[ty][0]
    wrap = (lambda x: x) if ty == 'f32' else (s32 if ty == 'i32' else u32)
    if space == 'device':
        body = ('%slot = and i32 %tid, 7\n'
                '%sp = getelementptr {t}, ptr addrspace(1) %out, i32 %slot\n'
                '%old = atomicrmw {op} ptr addrspace(1) %sp, {t} %a0 {o}\n').format(t=irty, op=op, o=ordering)
        full = None
    c = Case(name, [ty], ty, None, set(), None, ulps=0 if ty == 'f32' else None,
             source='LLVM atomicrmw (AIRLowerAtomicRMW)', out_init=[init] * N)
    expect = {}

    def ref(i, v, fn=fn, init=init, wrap=wrap):
        if 'r' not in expect:
            expect['r'] = fold_slots(v[0], fn, init, wrap)
        return expect['r'][i]
    c.ref = ref
    c.full = ('define void @{n}(ptr addrspace(1) %in0, ptr addrspace(1) %out) {{\n'
              'entry:\n'
              '  %t = call {{ i32, i32, i32 }} @air.thread_position_in_grid.v3i32()\n'
              '  %tid = extractvalue {{ i32, i32, i32 }} %t, 0\n'
              '  %p0 = getelementptr {t}, ptr addrspace(1) %in0, i32 %tid\n'
              '  %a0 = load {t}, ptr addrspace(1) %p0\n'
              '  ' + body.strip().replace('\n', '\n  ') + '\n'
              '  ret void\n}}').format(n=name, t=irty)
    return c


def group_atomics():
    """Atomics as front ends write them: atomicrmw, cmpxchg, atomic load/store."""
    import operator
    small = lambda i, r: r.randrange(0, 1000)
    cases = [
        atomic_case('atomic_add_i32', 'add', 'i32', operator.add),
        atomic_case('atomic_sub_i32', 'sub', 'i32', operator.sub),
        atomic_case('atomic_and_u32', 'and', 'u32', operator.and_, init=0xFFFFFFFF),
        atomic_case('atomic_or_u32', 'or', 'u32', operator.or_),
        atomic_case('atomic_xor_u32', 'xor', 'u32', operator.xor),
        atomic_case('atomic_max_i32', 'max', 'i32', max, init=-(1 << 31)),
        atomic_case('atomic_min_i32', 'min', 'i32', min, init=(1 << 31) - 1),
        atomic_case('atomic_umax_u32', 'umax', 'u32', max),
        atomic_case('atomic_umin_u32', 'umin', 'u32', min, init=0xFFFFFFFF),
        atomic_case('atomic_add_seq_cst_i32', 'add', 'i32', operator.add, ordering='seq_cst'),
    ]
    # Float add: small integers, so the sum is exact whatever the order.
    c = atomic_case('atomic_fadd_f32', 'fadd', 'f32', operator.add)
    c.gen = [lambda i, r: float(r.randrange(-100, 100))]
    cases.append(c)
    # xchg on the thread's own slot: the old value (0) comes back, the slot holds the input.
    cases.append(Case('atomic_xchg_i32', ['i32'], 'i32',
                      '%sp = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                      '%old = atomicrmw xchg ptr addrspace(1) %sp, i32 %a0 monotonic\n'
                      '%r0 = add i32 %old, %a0\n'
                      '%r = add i32 %r0, 0',
                      set(), lambda i, v: v[0][i], out_init=[0] * N))
    # Fix the xchg case: the template stores %r over the slot afterwards, which is the input as well.
    # cmpxchg: each thread swaps its own slot from tid to the input, and keeps whether it succeeded.
    cases.append(Case('cmpxchg_i32', ['i32'], 'i32',
                      '%sp = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                      '%pair = cmpxchg ptr addrspace(1) %sp, i32 %tid, i32 %a0 monotonic monotonic\n'
                      '%ok = extractvalue { i32, i1 } %pair, 1\n'
                      '%r = select i1 %ok, i32 %a0, i32 -1',
                      set(), lambda i, v: v[0][i], out_init=list(range(N))))
    # Atomic load and store.
    cases.append(Case('atomic_load_store_i32', ['i32'], 'i32',
                      '%la = load atomic i32, ptr addrspace(1) %p0 monotonic, align 4\n'
                      '%r = add i32 %la, 1',
                      set(), lambda i, v: s32(v[0][i] + 1)))
    cases.append(Case('fence_i32', ['i32'], 'i32',
                      'fence syncscope("workgroup") acq_rel\nfence seq_cst\n%r = add i32 %a0, 2',
                      set(), lambda i, v: s32(v[0][i] + 2)))
    return cases


def tg_kernel(name, body, decls=()):
    """A kernel over in0/out with a 64-element threadgroup array @tg_<name>."""
    return ('@tg_{n} = internal addrspace(3) global [64 x i32] undef, align 4\n'
            'define void @{n}(ptr addrspace(1) %in0, ptr addrspace(1) %out) {{\n'
            'entry:\n'
            '  %t = call {{ i32, i32, i32 }} @air.thread_position_in_grid.v3i32()\n'
            '  %tid = extractvalue {{ i32, i32, i32 }} %t, 0\n'
            '  %lt = call {{ i32, i32, i32 }} @air.thread_position_in_threadgroup.v3i32()\n'
            '  %lid = extractvalue {{ i32, i32, i32 }} %lt, 0\n'
            '  %p0 = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
            '  %a0 = load i32, ptr addrspace(1) %p0\n'
            '  %po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
            '  ' + body.strip().replace('\n', '\n  ').replace('@tg', '@tg_{n}') + '\n'
            '  ret void\n}}').format(n=name)


TG_DECLS = {'declare { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()',
            'declare void @air.wg.barrier(i32, i32)'}


def group_threadgroup():
    """Threadgroup memory, barriers and threadgroup atomics.

    Threadgroups of 64 threads are two SIMD-groups, so reading what another
    SIMD-group stored needs the barrier (one SIMD-group runs in lockstep and
    would pass without it).
    """
    rev = ('%sp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %lid\n'
           'store i32 %a0, ptr addrspace(3) %sp\n'
           'BARRIER\n'
           '%rl = sub i32 63, %lid\n'
           '%rp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %rl\n'
           '%r = load i32, ptr addrspace(3) %rp\n'
           'store i32 %r, ptr addrspace(1) %po')
    reverse = lambda i, v: v[0][63 - i]
    cases = []
    for name, barrier, decl in (
            ('tg_reverse_barrier2', 'call void @air.wg.barrier(i32 2, i32 1)',
             'declare void @air.wg.barrier(i32, i32)'),
            ('tg_reverse_barrier3', 'call void @air.wg.barrier(i32 2, i32 5, i32 1)',
             'declare void @air.wg.barrier(i32, i32, i32)')):
        c = Case(name, ['i32'], 'i32', None, {decl, 'declare { i32, i32, i32 } '
                 '@air.thread_position_in_threadgroup.v3i32()'}, reverse, tgsize=64,
                 source='floor metal.hpp (2 operands); Metal.jl synchronization.jl (3, AIR 2.9)')
        c.full = tg_kernel(name, rev.replace('BARRIER', barrier))
        cases.append(c)
    # A kernel with no explicit barrier: AIRTGBarrierInsert is meant to add one.
    c = Case('tg_reverse_auto_barrier', ['i32'], 'i32', None,
             {'declare { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()'},
             reverse, tgsize=64, source='AIRTGBarrierInsert')
    c.full = tg_kernel(c.name, rev.replace('BARRIER\n', ''))
    cases.append(c)
    # Threadgroup atomic: every thread adds its input; after a barrier each
    # thread reads the threadgroup total.
    c = Case('tg_atomic_add', ['i32'], 'i32', None, TG_DECLS,
             lambda i, v: s32(sum(v[0])), tgsize=64,
             gen=[lambda i, r: r.randrange(-1000, 1000)])
    c.full = tg_kernel(c.name, (
        '%z = icmp eq i32 %lid, 0\n'
        'br i1 %z, label %init, label %go\n'
        'init:\n'
        'store i32 0, ptr addrspace(3) @tg\n'
        'br label %go\n'
        'go:\n'
        'call void @air.wg.barrier(i32 2, i32 1)\n'
        '%old = atomicrmw add ptr addrspace(3) @tg, i32 %a0 monotonic\n'
        'call void @air.wg.barrier(i32 2, i32 1)\n'
        '%r = load i32, ptr addrspace(3) @tg\n'
        'store i32 %r, ptr addrspace(1) %po'))
    cases.append(c)
    return cases


# (role, vector?, value for thread i of a 64-thread grid in threadgroups of 32)
SYSVALS = [
    ('thread_position_in_grid', True, lambda i: i),
    ('thread_position_in_threadgroup', True, lambda i: i % 32),
    ('threadgroup_position_in_grid', True, lambda i: i // 32),
    ('threads_per_threadgroup', True, lambda i: 32),
    ('threadgroups_per_grid', True, lambda i: 2),
    ('threads_per_grid', True, lambda i: 64),
    ('thread_index_in_threadgroup', False, lambda i: i % 32),
    ('thread_index_in_simdgroup', False, lambda i: i % 32),
    ('simdgroup_index_in_threadgroup', False, lambda i: 0),
    ('threads_per_simdgroup', False, lambda i: 32),
    ('simdgroups_per_threadgroup', False, lambda i: 1),
]


def group_sysvals():
    """Every system value, read as a call and written out (x component)."""
    cases = []
    for role, vec, ref in SYSVALS:
        if vec:
            decl = 'declare { i32, i32, i32 } @air.%s.v3i32()' % role
            body = ('%%s = call { i32, i32, i32 } @air.%s.v3i32()\n'
                    '%%r = extractvalue { i32, i32, i32 } %%s, 0') % role
        else:
            decl = 'declare i32 @air.%s.i32()' % role
            body = '%%r = call i32 @air.%s.i32()' % role
        cases.append(Case('sv_' + role, ['u32'], 'u32', body, {decl},
                          lambda i, v, ref=ref: ref(i), tgsize=32,
                          source='Metal attribute [[%s]]' % role))
    return cases


def full_case(name, ins, out, ref, body, decls=(), gen=None, ulps=None, abstol=None, source=''):
    """A kernel with a hand-written body. %in0.. and %out are the buffers,
    %tid the thread; the body must store out[tid] itself."""
    args = ', '.join(['ptr addrspace(1) %%in%d' % i for i in range(len(ins))] +
                     ['ptr addrspace(1) %out'])
    c = Case(name, ins, out, None, set(decls), ref, gen, ulps, source)
    c.abstol = abstol
    c.full = ('define void @%s(%s) {\nentry:\n'
              '  %%t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()\n'
              '  %%tid = extractvalue { i32, i32, i32 } %%t, 0\n  ' % (name, args) +
              body.strip().replace('\n', '\n  ') + '\n}')
    return c


def s16(x):
    return ((x + (1 << 15)) % (1 << 16)) - (1 << 15)


def s8(x):
    return ((x + (1 << 7)) % (1 << 8)) - (1 << 7)


def s64(x):
    return ((x + (1 << 63)) % (1 << 64)) - (1 << 63)


def group_types():
    """Half, vectors, narrow and wide integers, conversions."""
    small = lambda i, r: r.uniform(-4, 4)
    cases = []
    # half arithmetic and math (inputs and outputs are half).
    for name, op, ref in (('f16_fmul', 'fmul half %a, %b', lambda a, b: a * b),
                          ('f16_fadd', 'fadd half %a, %b', lambda a, b: a + b)):
        c = full_case(name, ['f16', 'f16'], 'f16', lambda i, v, ref=ref: f16(ref(v[0][i], v[1][i])),
                      '%pa = getelementptr half, ptr addrspace(1) %in0, i32 %tid\n'
                      '%a = load half, ptr addrspace(1) %pa\n'
                      '%pb = getelementptr half, ptr addrspace(1) %in1, i32 %tid\n'
                      '%b = load half, ptr addrspace(1) %pb\n'
                      '%r = ' + op + '\n'
                      '%po = getelementptr half, ptr addrspace(1) %out, i32 %tid\n'
                      'store half %r, ptr addrspace(1) %po\nret void',
                      gen=[small, small])
        c.ulps = None
        c.abstol = 0.0
        cases.append(c)
    for fn, ref in (('sin', math.sin), ('sqrt', math.sqrt), ('exp2', lambda x: 2.0 ** x)):
        gen = (lambda i, r: r.uniform(0.01, 8)) if fn == 'sqrt' else small
        c = full_case('f16_llvm_' + fn, ['f16'], 'f16', lambda i, v, ref=ref: f16(ref(v[0][i])),
                      ('%pa = getelementptr half, ptr addrspace(1) %in0, i32 %tid\n'
                       '%a = load half, ptr addrspace(1) %pa\n'
                       '%r = call half @llvm.FN.f16(half %a)\n'
                       '%po = getelementptr half, ptr addrspace(1) %out, i32 %tid\n'
                       'store half %r, ptr addrspace(1) %po\nret void').replace('FN', fn),
                      {'declare half @llvm.%s.f16(half)' % fn}, gen=[gen])
        c.ulps = None
        c.abstol = 2.0 ** -8 * 8  # half has 11 bits; allow a few ulps at |x| <= 8
        cases.append(c)
    # <4 x float>: vector loads, vector math, vector stores (thread i handles 4 floats).
    for fn, ref in (('sin', math.sin), ('fabs', abs), ('floor', math.floor)):
        def vref(i, v, ref=ref):
            return ref(v[0][i])
        c = full_case('v4f32_llvm_' + fn, ['f32'], 'f32', vref,
                      '%q = and i32 %tid, 15\n'
                      '%is = icmp eq i32 %tid, %q\n'
                      'br i1 %is, label %go, label %done\n'
                      'go:\n'
                      '%pa = getelementptr <4 x float>, ptr addrspace(1) %in0, i32 %tid\n'
                      '%a = load <4 x float>, ptr addrspace(1) %pa\n'
                      '%r = call <4 x float> @llvm.' + fn + '.v4f32(<4 x float> %a)\n'
                      '%po = getelementptr <4 x float>, ptr addrspace(1) %out, i32 %tid\n'
                      'store <4 x float> %r, ptr addrspace(1) %po\n'
                      'br label %done\n'
                      'done:\nret void',
                      {'declare <4 x float> @llvm.%s.v4f32(<4 x float>)' % fn},
                      ulps=0 if fn != 'sin' else 4)
        cases.append(c)
    # Narrow and wide integers in memory and arithmetic.
    for ty, w, irty in (('i8', s8, 'i8'), ('i16', s16, 'i16'), ('i64', s64, 'i64')):
        c = full_case('int_' + ty + '_muladd', [ty, ty], ty,
                      lambda i, v, w=w: w(v[0][i] * v[1][i] + 3),
                      '%pa = getelementptr {t}, ptr addrspace(1) %in0, i32 %tid\n'
                      '%a = load {t}, ptr addrspace(1) %pa\n'
                      '%pb = getelementptr {t}, ptr addrspace(1) %in1, i32 %tid\n'
                      '%b = load {t}, ptr addrspace(1) %pb\n'
                      '%m = mul {t} %a, %b\n%r = add {t} %m, 3\n'
                      '%po = getelementptr {t}, ptr addrspace(1) %out, i32 %tid\n'
                      'store {t} %r, ptr addrspace(1) %po\nret void'.format(t=irty))
        cases.append(c)
    c = full_case('int_i64_udiv', ['u64', 'u64'], 'u64',
                  lambda i, v: v[0][i] // v[1][i],
                  '%pa = getelementptr i64, ptr addrspace(1) %in0, i32 %tid\n'
                  '%a = load i64, ptr addrspace(1) %pa\n'
                  '%pb = getelementptr i64, ptr addrspace(1) %in1, i32 %tid\n'
                  '%b = load i64, ptr addrspace(1) %pb\n'
                  '%r = udiv i64 %a, %b\n'
                  '%po = getelementptr i64, ptr addrspace(1) %out, i32 %tid\n'
                  'store i64 %r, ptr addrspace(1) %po\nret void',
                  gen=[None, lambda i, r: r.randrange(1, 1 << 40)])
    cases.append(c)
    c = full_case('int_i32_sdiv_srem', ['i32', 'i32'], 'i32',
                  lambda i, v: s32(int(v[0][i] / v[1][i]) * 7 + int(math.fmod(v[0][i], v[1][i]))),
                  '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
                  '%a = load i32, ptr addrspace(1) %pa\n'
                  '%pb = getelementptr i32, ptr addrspace(1) %in1, i32 %tid\n'
                  '%b = load i32, ptr addrspace(1) %pb\n'
                  '%q = sdiv i32 %a, %b\n%m = srem i32 %a, %b\n%q7 = mul i32 %q, 7\n'
                  '%r = add i32 %q7, %m\n'
                  '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                  'store i32 %r, ptr addrspace(1) %po\nret void',
                  gen=[lambda i, r: r.randrange(-100000, 100000), lambda i, r: r.choice([-7, -3, 1, 5, 13, 1000])])
    cases.append(c)
    # Conversions.
    conv = [
        ('cvt_sitofp', 'i32', 'f32', 'sitofp i32 %a to float', lambda x: float(x), None),
        ('cvt_uitofp', 'u32', 'f32', 'uitofp i32 %a to float', lambda x: float(x), None),
        ('cvt_fptosi', 'f32', 'i32', 'fptosi float %a to i32', lambda x: int(x),
         lambda i, r: r.uniform(-1e6, 1e6)),
        ('cvt_fptoui', 'f32', 'u32', 'fptoui float %a to i32', lambda x: int(x),
         lambda i, r: r.uniform(0, 4e9)),
        ('cvt_fptrunc', 'f32', 'f16', 'fptrunc float %a to half', f16, small),
        ('cvt_fpext', 'f16', 'f32', 'fpext half %a to float', lambda x: x, small),
        ('cvt_sext8', 'i8', 'i32', 'sext i8 %a to i32', lambda x: x, None),
        ('cvt_zext16', 'u16', 'u32', 'zext i16 %a to i32', lambda x: x, None),
        ('cvt_trunc64', 'i64', 'i32', 'trunc i64 %a to i32', s32, None),
        ('cvt_sitofp64', 'i64', 'f32', 'sitofp i64 %a to float', lambda x: float(x), None),
    ]
    for name, ti, to, inst, ref, gen in conv:
        c = full_case(name, [ti], to, lambda i, v, ref=ref: ref(v[0][i]),
                      '%pa = getelementptr {ti}, ptr addrspace(1) %in0, i32 %tid\n'
                      '%a = load {ti}, ptr addrspace(1) %pa\n'
                      '%r = {inst}\n'
                      '%po = getelementptr {to}, ptr addrspace(1) %out, i32 %tid\n'
                      'store {to} %r, ptr addrspace(1) %po\nret void'.format(
                          ti=TYPES[ti][0], to=TYPES[to][0], inst=inst),
                      gen=[gen])
        if to in ('f32', 'f16'):
            c.ulps = 0 if to == 'f32' else None
            c.abstol = 0.0 if to == 'f16' else None
        cases.append(c)
    return cases


def group_memory():
    """Aggregates, pointer phis and selects, private and constant memory, loops."""
    cases = []
    # Struct GEPs chosen through a pointer select (32 structs of two i32).
    cases.append(full_case(
        'mem_struct_select', ['u32'], 'u32', lambda i, v: v[0][2 * (i % 32) + i // 32],
        '%idx = and i32 %tid, 31\n'
        '%hi = icmp uge i32 %tid, 32\n'
        '%f0 = getelementptr { i32, i32 }, ptr addrspace(1) %in0, i32 %idx, i32 0\n'
        '%f1 = getelementptr { i32, i32 }, ptr addrspace(1) %in0, i32 %idx, i32 1\n'
        '%fp = select i1 %hi, ptr addrspace(1) %f1, ptr addrspace(1) %f0\n'
        '%r = load i32, ptr addrspace(1) %fp\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='typed-pointer reconstruction'))
    # Nested aggregate {i32, [3 x i32]} reached through a pointer phi.
    cases.append(full_case(
        'mem_nested_phi', ['u32'], 'u32', lambda i, v: v[0][4 * (i % 16) + i // 16],
        '%e = and i32 %tid, 15\n'
        '%j = lshr i32 %tid, 4\n'
        '%first = icmp eq i32 %j, 0\n'
        'br i1 %first, label %head, label %tail\n'
        'head:\n'
        '%ph = getelementptr { i32, [3 x i32] }, ptr addrspace(1) %in0, i32 %e, i32 0\n'
        'br label %join\n'
        'tail:\n'
        '%k = sub i32 %j, 1\n'
        '%pt = getelementptr { i32, [3 x i32] }, ptr addrspace(1) %in0, i32 %e, i32 1, i32 %k\n'
        'br label %join\n'
        'join:\n'
        '%pp = phi ptr addrspace(1) [ %ph, %head ], [ %pt, %tail ]\n'
        '%r = load i32, ptr addrspace(1) %pp\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='typed-pointer reconstruction'))
    # Private array with dynamic indices.
    cases.append(full_case(
        'mem_private_array', ['i32'], 'i32', lambda i, v: s32(v[0][i] + i % 8 * 3),
        '%arr = alloca [8 x i32]\n'
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = load i32, ptr addrspace(1) %pa\n'
        'br label %fill\n'
        'fill:\n'
        '%k = phi i32 [ 0, %entry ], [ %k1, %fill ]\n'
        '%k3 = mul i32 %k, 3\n%v = add i32 %a, %k3\n'
        '%slot = getelementptr [8 x i32], ptr %arr, i32 0, i32 %k\n'
        'store i32 %v, ptr %slot\n'
        '%k1 = add i32 %k, 1\n%more = icmp ult i32 %k1, 8\n'
        'br i1 %more, label %fill, label %read\n'
        'read:\n'
        '%sel = and i32 %tid, 7\n'
        '%sp = getelementptr [8 x i32], ptr %arr, i32 0, i32 %sel\n'
        '%r = load i32, ptr %sp\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='private memory'))
    # Loop whose trip count differs per thread.
    cases.append(full_case(
        'mem_divergent_loop', ['u32'], 'u32', lambda i, v: sum(range(v[0][i] % 16)),
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = load i32, ptr addrspace(1) %pa\n'
        '%n = and i32 %a, 15\n'
        '%z = icmp eq i32 %n, 0\n'
        'br i1 %z, label %done, label %loop\n'
        'loop:\n'
        '%k = phi i32 [ 0, %entry ], [ %k1, %loop ]\n'
        '%acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]\n'
        '%acc1 = add i32 %acc, %k\n%k1 = add i32 %k, 1\n'
        '%more = icmp ult i32 %k1, %n\n'
        'br i1 %more, label %loop, label %done\n'
        'done:\n'
        '%r = phi i32 [ 0, %entry ], [ %acc1, %loop ]\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='control flow'))
    # Two buffers chosen through a pointer select on the thread's parity.
    cases.append(full_case(
        'mem_buffer_select', ['i32', 'i32'], 'i32', lambda i, v: v[i % 2][i],
        '%odd = trunc i32 %tid to i1\n'
        '%b = select i1 %odd, ptr addrspace(1) %in1, ptr addrspace(1) %in0\n'
        '%pb = getelementptr i32, ptr addrspace(1) %b, i32 %tid\n'
        '%r = load i32, ptr addrspace(1) %pb\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='typed-pointer reconstruction'))
    # A constant lookup table in the constant address space.
    lut = [3, 1, 4, 1, 5, 9, 2, 6]
    c = full_case(
        'mem_constant_table', ['u32'], 'u32', lambda i, v: lut[v[0][i] % 8] + 1,
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = load i32, ptr addrspace(1) %pa\n'
        '%k = and i32 %a, 7\n'
        '%lp = getelementptr [8 x i32], ptr addrspace(2) @lut_table, i32 0, i32 %k\n'
        '%l = load i32, ptr addrspace(2) %lp\n%r = add i32 %l, 1\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='constant address space')
    c.full = ('@lut_table = internal addrspace(2) constant [8 x i32] [%s]\n' %
              ', '.join('i32 %d' % x for x in lut)) + c.full
    cases.append(c)
    # A constant-address-space buffer argument.
    c = full_case(
        'mem_constant_buffer', ['i32'], 'i32', lambda i, v: s32(v[0][i] * 5),
        '%pa = getelementptr i32, ptr addrspace(2) %in0, i32 %tid\n'
        '%a = load i32, ptr addrspace(2) %pa\n%r = mul i32 %a, 5\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void', source='constant address space')
    c.full = c.full.replace('ptr addrspace(1) %in0', 'ptr addrspace(2) %in0', 1)
    cases.append(c)
    # llvm.memcpy from device to private and back.
    cases.append(full_case(
        'mem_memcpy', ['i32'], 'i32', lambda i, v: v[0][i],
        '%tmp = alloca [4 x i32]\n'
        '%q = and i32 %tid, 15\n%base = shl i32 %q, 2\n'
        '%is = icmp eq i32 %tid, %q\n'
        'br i1 %is, label %go, label %done\n'
        'go:\n'
        '%src = getelementptr i32, ptr addrspace(1) %in0, i32 %base\n'
        'call void @llvm.memcpy.p0.p1.i64(ptr %tmp, ptr addrspace(1) %src, i64 16, i1 false)\n'
        '%dst = getelementptr i32, ptr addrspace(1) %out, i32 %base\n'
        'call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) %dst, ptr %tmp, i64 16, i1 false)\n'
        'br label %done\n'
        'done:\nret void',
        {'declare void @llvm.memcpy.p0.p1.i64(ptr, ptr addrspace(1), i64, i1)',
         'declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1), ptr, i64, i1)'},
        source='floor MetalFinal: memcpy is accepted'))
    # Scalar kernel arguments, which the backend packs into a buffer after out.
    c = Case('mem_scalar_args', [], 'f32', None, set(), lambda i, v: f32(7 * 2.5 + i),
             scalars=[('i32', 7), ('f32', 2.5)], ulps=0, source='AIRScalarBufferPacking')
    c.full = ('define void @mem_scalar_args(ptr addrspace(1) %out, i32 %n, float %s) {\n'
              'entry:\n'
              '  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()\n'
              '  %tid = extractvalue { i32, i32, i32 } %t, 0\n'
              '  %f = sitofp i32 %n to float\n  %v = fmul float %f, %s\n'
              '  %ft = uitofp i32 %tid to float\n  %r = fadd float %v, %ft\n'
              '  %p = getelementptr float, ptr addrspace(1) %out, i32 %tid\n'
              '  store float %r, ptr addrspace(1) %p\n  ret void\n}')
    cases.append(c)
    return cases


def matmul(X, Y, tx=False):
    """Row-major 8x8 product; tx transposes X first."""
    at = (lambda r, k: X[k * 8 + r]) if tx else (lambda r, k: X[r * 8 + k])
    return [sum(at(r, k) * Y[k * 8 + c] for k in range(8)) for r in range(8) for c in range(8)]


def matrix_case(name, ty, a_strides, c_init, tx, tg=False):
    t, sfx = ('float', 'f32') if ty == 'f32' else ('half', 'f16')
    asp = 3 if tg else 1
    vt = '<64 x %s>' % t
    load = 'air.simdgroup_matrix_8x8_load.v64%s.p%d%s' % (sfx, asp, sfx)
    store = 'air.simdgroup_matrix_8x8_store.v64%s.p1%s' % (sfx, sfx)
    mma = 'air.simdgroup_matrix_8x8_multiply_accumulate.v64{0}.v64{0}.v64{0}.v64{0}'.format(sfx)
    fill = 'air.simdgroup_matrix_8x8_init_filled.v64%s.%s' % (sfx, sfx)
    rs = '<2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer'
    as_ = '<2 x i64> <i64 8, i64 8>, <2 x i64> <i64 %d, i64 %d>, <2 x i64> zeroinitializer' % a_strides
    decls = {'declare %s @%s(ptr addrspace(%d), <2 x i64>, <2 x i64>, <2 x i64>)' % (vt, load, asp),
             'declare void @%s(%s, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)' % (store, vt),
             'declare %s @%s(%s, %s, %s)' % (vt, mma, vt, vt, vt),
             'declare %s @%s(%s)' % (vt, fill, t)}
    if tg:
        decls |= TG_DECLS
        stage = ('@mtg_a_{n} = internal addrspace(3) global [64 x {t}] undef, align 4\n'
                 '@mtg_b_{n} = internal addrspace(3) global [64 x {t}] undef, align 4\n').format(n=name, t=t)
        copy = ('  %lt = call { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()\n'
                '  %lid = extractvalue { i32, i32, i32 } %lt, 0\n'
                '  %l2 = shl i32 %lid, 1\n'
                + ''.join(
                    '  %%sa{k} = getelementptr {t}, ptr addrspace(1) %%in0, i32 %%i{k}\n'
                    '  %%va{k} = load {t}, ptr addrspace(1) %%sa{k}\n'
                    '  %%da{k} = getelementptr [64 x {t}], ptr addrspace(3) @mtg_a_{n}, i32 0, i32 %%i{k}\n'
                    '  store {t} %%va{k}, ptr addrspace(3) %%da{k}\n'
                    '  %%sb{k} = getelementptr {t}, ptr addrspace(1) %%in1, i32 %%i{k}\n'
                    '  %%vb{k} = load {t}, ptr addrspace(1) %%sb{k}\n'
                    '  %%db{k} = getelementptr [64 x {t}], ptr addrspace(3) @mtg_b_{n}, i32 0, i32 %%i{k}\n'
                    '  store {t} %%vb{k}, ptr addrspace(3) %%db{k}\n'.format(k=k, t=t, n=name)
                    .replace('%%', '%') for k in (0, 1))
                .replace('%i0', '%l2').replace('%i1', '%l2p')
                .replace('  %sa0', '  %l2p = add i32 %l2, 1\n  %sa0', 1)
                + '  call void @air.wg.barrier(i32 2, i32 1)\n')
        pa, pb = '@mtg_a_%s' % name, '@mtg_b_%s' % name
    else:
        stage, copy, pa, pb = '', '', '%in0', '%in1'
    if c_init:
        cinit = '  %mc = call {vt} @{fill}({t} 0.0)\n  %mc1 = fadd {vt} %mc, splat ({t} 2.0)\n'.format(vt=vt, fill=fill, t=t)
        cname = '%mc1'
    else:
        cinit = '  %mc = call {vt} @{fill}({t} 0.0)\n'.format(vt=vt, fill=fill, t=t)
        cname = '%mc'
    body = (stage +
            'define void @{n}(ptr addrspace(1) %in0, ptr addrspace(1) %in1, ptr addrspace(1) %out) {{\n'
            'entry:\n'.format(n=name) + copy +
            '  %ma = call {vt} @{load}(ptr addrspace({asp}) {pa}, {as_})\n'
            '  %mb = call {vt} @{load}(ptr addrspace({asp}) {pb}, {rs})\n'.format(
                vt=vt, load=load, asp=asp, pa=pa, pb=pb, as_=as_, rs=rs) + cinit +
            '  %r = call {vt} @{mma}({vt} %ma, {vt} %mb, {vt} {c})\n'
            '  call void @{store}({vt} %r, ptr addrspace(1) %out, {rs})\n'
            '  ret void\n}}'.format(vt=vt, mma=mma, c=cname, store=store, rs=rs))
    small = lambda i, r: float(r.randrange(-4, 5))
    extra = 2.0 if c_init else 0.0
    c = Case(name, [ty, ty], ty, None, decls,
             lambda i, v: matmul(v[0], v[1], tx)[i] + extra, [small, small],
             0 if ty == 'f32' else None, 'Metal.jl simd.jl (AIR 2.8 matrix signatures)',
             tgsize=32)
    if ty == 'f16':
        c.abstol = 0.0
    c.full = body
    return c


def group_matrix():
    """SIMD-group 8x8 matrices: load, multiply-accumulate, store."""
    return [
        matrix_case('mma_f32', 'f32', (1, 8), False, False),
        matrix_case('mma_f32_acc', 'f32', (1, 8), True, False),
        matrix_case('mma_f32_transposed_a', 'f32', (8, 1), False, True),
        matrix_case('mma_f16', 'f16', (1, 8), False, False),
        matrix_case('mma_f32_threadgroup', 'f32', (1, 8), False, False, tg=True),
    ]


def group_misc():
    """Device-coherent memory, mul_hi, the AIR 2.9 atomic form."""
    cases = []
    cases.append(full_case(
        'device_coherent_load_store', ['i32'], 'i32', lambda i, v: s32(v[0][i] + 9),
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = call i32 @air.load.device_coherent.i32.p1i32(ptr addrspace(1) %pa)\n'
        '%r = add i32 %a, 9\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'call void @air.store.device_coherent.i32.p1i32(i32 %r, ptr addrspace(1) %po)\nret void',
        {'declare i32 @air.load.device_coherent.i32.p1i32(ptr addrspace(1))',
         'declare void @air.store.device_coherent.i32.p1i32(i32, ptr addrspace(1))'},
        source='dxmt CreateDeviceCoherentLoad/Store'))
    for sfx, ty, ref in (('s.i32', 'i32', lambda a, b: (a * b) >> 32),
                         ('u.i32', 'u32', lambda a, b: (a * b) >> 32)):
        cases.append(int_case('air_mul_hi_' + sfx.replace('.', '_'), 'air.mul_hi.' + sfx, ty, 2,
                              lambda i, v, ref=ref: ref(v[0][i], v[1][i])))
    # AIR 2.9 compare-and-exchange, with memory flags (8 operands); the
    # backend raises the module's AIR version for it.
    c = full_case(
        'cmpxchg_air29_flags', ['i32'], 'i32', lambda i, v: v[0][i],
        '%exp = alloca i32\n'
        'store i32 %tid, ptr %exp\n'
        '%sp = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        '%old = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %sp, ptr %exp, i32 %a0, i32 0, i32 0, i32 2, i32 1, i1 true)\n'
        'ret void',
        {'declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i32, i1)'},
        source='floor metal_atomic.hpp (MSL 4.1 flags operand)')
    c.host_limit = 'needs MSL 4.1 (AIR 2.9)'
    c.full = c.full.replace('entry:\n', 'entry:\n  %p0 = getelementptr i32, ptr addrspace(1) %in0, i32 0\n', 1)
    c.full = c.full.replace('%exp = alloca i32', '%pa0 = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n  %a0 = load i32, ptr addrspace(1) %pa0\n  %exp = alloca i32')
    c.out_init = list(range(N))
    cases.append(c)
    return cases


GROUPS = {
    'llvm-math-f32': group_llvm_math_f32,
    'air-math-f32': group_air_math_f32,
    'llvm-math-afn': group_llvm_math_afn,
    'llvm-int': group_llvm_int,
    'simd': group_simd,
    'atomics': group_atomics,
    'threadgroup': group_threadgroup,
    'sysvals': group_sysvals,
    'types': group_types,
    'memory': group_memory,
    'matrix': group_matrix,
    'misc': group_misc,
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
    return 1 if set(counts) - {'PASS', 'ABSENT', 'HOST-LIMIT'} else 0


if __name__ == '__main__':
    sys.exit(main())
