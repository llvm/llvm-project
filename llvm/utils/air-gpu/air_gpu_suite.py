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
  HW-FTZ    the device ran the kernel and every wrong result is explained by
            the hardware flushing float32 subnormals to zero: each wrong thread
            has a subnormal float32 input or a subnormal exact result, and its
            value is the reference computed with float32 subnormals flushed
            (inputs and result; the device value flushed too). Apple GPUs
            flush float32 subnormals, a hardware limit, not a backend bug
  COMPILE   llc failed
  ABSENT    the device rejected a probe for a function AIR does not have, as
            expected (these record what the catalogue must not emit)
  SKIPPED   (AGX only) the case relies on a pass only the Metal pipeline runs

  air_gpu_suite.py --llc build/bin/llc [--group math-f32 ...] [--json out.json]

A case runs N = 64 threads, by default as a 1D grid in threadgroups of
tgsize; a case can set grid and group (x, y, z) for a 2D or 3D dispatch of
the same 64 threads (dispatch-3d).

With --agx-run, the suite runs the same kernels on AGX instead: each case is
compiled by the AIR target's portable passes (llc -air-portable) and the AGX
backend (llc -mtriple=agx, whose agx-lower-air pass reads that AIR), and run
on the AGX Performance Simulator's agx_run, through
llvm/utils/agx-checked/run_checked.py, against the same references. COMPILE
then also covers what agx-lower-air does not lower.

  air_gpu_suite.py --llc build/bin/llc --agx-run agx_run \
      --objcopy build/bin/llvm-objcopy --readelf build/bin/llvm-readelf

With --compare-backends (and --agx-run), every case runs on both, and the two
outputs are compared with each other bit for bit (the references are still
computed, and each side's verdict reported). A case is SAME or DIFF-<class>,
from the most severe differing thread: ftz (equal once float32 subnormals are
flushed), nan (both NaNs), ulp (a case with a tolerance, within 16 ULPs or
its own ULP tolerance; the distance is reported) or real; NOT-COMPARED when a side did not run it.

Requests to the service are sent one at a time under the device lock
(/scratch/air-device.lock, AIR_DEVICE_LOCK), named in --progress before they
are sent. A request that times out is never resent: it is added to
air_gpu_suite_hangs.txt (never sent again) and all device work stops.
"""

import argparse
import base64
import hashlib
import functools
from fractions import Fraction
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile



# Every tool sending requests to the AIR service takes this lock around each one, so that requests never overlap (a
# request the Mac never finishes freezes its display; two at once make that worse and hide which one it was).
DEVICE_LOCK = os.environ.get('AIR_DEVICE_LOCK', '/scratch/air-device.lock')
# --progress: each request is named in this file before it is sent, so that a request that hangs is known.
PROGRESS = None
# Requests that hang the Mac ("GROUP CASE" lines, CASE * for a whole group library): never sent.
HANGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'air_gpu_suite_hangs.txt')
HANGS = set()
HANG_ERRORS = ('did not reply', 'timed out', 'timeout', 'time out')


class DeviceHang(Exception):
    """A request did not finish on the device: all device work stops, and it is never resent."""


def load_hangs():
    if os.path.exists(HANGS_FILE):
        for line in open(HANGS_FILE):
            line = line.split('#')[0].split()
            if len(line) == 2:
                HANGS.add(tuple(line))


def submit(job, queue, timeout, label='?'):
    import fcntl
    sys.path.insert(0, os.environ.get('AIR_CLIENT', '/air/client'))
    from client import submit as service_submit
    with open(DEVICE_LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if PROGRESS:
            with open(PROGRESS, 'a') as f:
                f.write('%s %s\n' % (label, job.get('op')))
                f.flush()
                os.fsync(f.fileno())
        return service_submit(job, queue, timeout=timeout)

N = 64  # threads per case: two SIMD-groups

TYPES = {
    'f32': ('float', 'f', 4), 'f16': ('half', 'e', 2),
    'i32': ('i32', 'i', 4), 'u32': ('i32', 'I', 4),
    'i16': ('i16', 'h', 2), 'u16': ('i16', 'H', 2),
    'i8': ('i8', 'b', 1), 'u8': ('i8', 'B', 1),
    'i64': ('i64', 'q', 8), 'u64': ('i64', 'Q', 8),
    'i1': ('i1', 'B', 1),
    'bf16': ('bfloat', 'H', 2),  # values are the bit patterns
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


def float_bits(ty, x):
    """The encoding of x in the float type ty (f32 or f16)."""
    return struct.pack('<f', f32(x)) if ty == 'f32' else struct.pack('<e', f16(x))


def int_to_f32(x):
    """The integer x rounded to float32, to nearest even, without the double
    rounding of float(x) for integers wider than 53 bits."""
    m = abs(x)
    n = m.bit_length()
    if n > 24:
        sh = n - 24
        q, rem, half = m >> sh, m & ((1 << sh) - 1), 1 << (sh - 1)
        if rem > half or (rem == half and q & 1):
            q += 1
        m = q << sh
    return -float(m) if x < 0 else float(m)


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
                 scalars=None, host_limit=None, metal_only=None):
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
        # Depends on a pass that only the Metal pipeline runs (a reason):
        # skipped on AGX.
        self.metal_only = metal_only
        # Floats: compare encodings (the sign of zero), any NaN matching any NaN.
        self.bitexact = False
        # A mismatch explained by the hardware flushing float32 subnormals to
        # zero is reported as HW-FTZ (see hw_ftz); False keeps it a MISMATCH.
        self.ftz_ok = True
        # Dispatch shape: threads per grid and per threadgroup (x, y, z). The
        # default is a 1D grid of N threads in threadgroups of tgsize.
        self.grid = None
        self.group = None


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
        err = r.stderr.strip()
        lines = [l for l in err.splitlines() if 'error' in l.lower()]
        return None, (lines[0] if lines else err)[-400:] or 'llc exited with %d' % r.returncode
    data = open(air, 'rb').read()
    # Kernels with emulated atomics (AIRLowerAtomicRMW) take a lock table at a location index of its own.
    dis = subprocess.run([os.path.join(os.path.dirname(llc), 'llvm-dis'), air, '-o', '-'],
                         capture_output=True, text=True).stdout
    if '!"air_atomic_locks"' in dis:
        LOCK_TABLE_AIRS.add(hashlib.sha1(data).hexdigest())
    return data, None


# The lock table of emulated atomics (llvm/lib/Target/AIR/AIRLowerAtomicRMW.h): location index and size of the
# zeroed buffer bound for the compiled modules (sha1 of their AIR) that have one.
LOCK_TABLE_INDEX, LOCK_TABLE_BYTES = 30, 4096
LOCK_TABLE_AIRS = set()


def lock_table(job, air):
    """Bind a zeroed lock table to every command of job when the module air has one."""
    if hashlib.sha1(air).hexdigest() not in LOCK_TABLE_AIRS:
        return job
    b = len(job['buffers'])
    job['buffers'].append({'size': LOCK_TABLE_BYTES, 'data': base64.b64encode(bytes(LOCK_TABLE_BYTES)).decode()})
    for cmd in job['commands']:
        cmd['bindings'].append({'index': LOCK_TABLE_INDEX, 'buffer': b})
    return job


TRANSIENT = ('XPC_ERROR_CONNECTION_INTERRUPTED', 'did not reply')


def call(job, queue, tries=4, label='?'):
    """submit(), retrying the service's transient failures. A request that times out is not resent: it is added
    to the hang list and DeviceHang stops all device work."""
    if tuple(label.split()) in HANGS:
        return {'ok': False, 'error': 'on the hang list (%s), not sent' % os.path.basename(HANGS_FILE)}
    for attempt in range(tries):
        try:
            reply = submit(job, queue, timeout=120, label=label)
        except TimeoutError as e:
            reply = {'ok': False, 'error': str(e)}
        err = reply.get('error', '')
        if not reply.get('ok') and any(t in err.lower() for t in HANG_ERRORS):
            with open(HANGS_FILE, 'a') as f:
                f.write('# request timed out: %s\n%s\n' % (err[:120].replace('\n', ' '), label))
            raise DeviceHang('%s: %s' % (label, err[:300]))
        if reply.get('ok') or not any(t in err for t in TRANSIENT):
            return reply
    return reply


def dispatch(c):
    """The case's threads per grid and per threadgroup."""
    return list(c.grid or (N, 1, 1)), list(c.group or (c.tgsize, 1, 1))


def run_case(c, air, inputs, queue, label='?'):
    if c.scalars is not None:
        out = {'size': N * TYPES[c.out][2]}
        packed = b''.join(struct.pack('<' + TYPES[t][1], v) for t, v in c.scalars)
        buffers = [out, {'size': len(packed), 'data': base64.b64encode(packed).decode()}]
        job = {'op': 'execute', 'air': base64.b64encode(air).decode(), 'buffers': buffers,
               'commands': [{'kernel': c.name,
                             'bindings': [{'index': i, 'buffer': i} for i in range(2)],
                             'grid': dispatch(c)[0], 'group': dispatch(c)[1]}],
               'readback': [0]}
        return call(lock_table(job, air), queue, label=label)
    buffers = [{'size': len(d), 'data': base64.b64encode(d).decode()} for d in inputs]
    out = {'size': N * TYPES[c.out][2]}
    if c.out_init is not None:
        out['data'] = base64.b64encode(pack(c.out, c.out_init)).decode()
    buffers.append(out)
    job = {'op': 'execute', 'air': base64.b64encode(air).decode(), 'buffers': buffers,
           'commands': [{'kernel': c.name,
                         'bindings': [{'index': i, 'buffer': i} for i in range(len(buffers))],
                         'grid': dispatch(c)[0], 'group': dispatch(c)[1]}],
           'readback': [len(buffers) - 1]}
    return call(lock_table(job, air), queue, label=label)


def check(c, values, got):
    want = [c.ref(i, values) for i in range(N)]
    worst, bad = 0, []
    for i, (g, w) in enumerate(zip(got, want)):
        if c.bitexact:
            if not ((g != g and w != w) or float_bits(c.out, g) == float_bits(c.out, w)):
                bad.append((i, g, w))
        elif c.ulps is None and c.abstol is not None:
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


def compile_agx(llc, ir, tmp, tag, mcpu):
    """The case through the AIR target's portable passes and the AGX backend."""
    ll = os.path.join(tmp, tag + '.ll')
    portable = os.path.join(tmp, tag + '.portable.ll')
    obj = os.path.join(tmp, tag + '.o')
    open(ll, 'w').write(ir)
    for cmd in ([llc, '-mtriple=air', '-air-portable', ll, '-o', portable],
                [llc, '-mtriple=agx', '-mcpu=' + mcpu, '-filetype=obj', portable, '-o', obj]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr.strip()[-400:]
    return obj, None


def scalar_field(ty, v):
    """A scalar argument as a field of run_checked's 'param' argument."""
    if TYPES[ty][2] == 8:
        return ('ptr', struct.unpack('<Q', struct.pack('<' + TYPES[ty][1], v))[0])
    return ('i32', struct.unpack('<I', struct.pack('<' + TYPES[ty][1], v).ljust(4, b'\0'))[0])


def run_case_agx(c, obj, inputs, agx):
    """Returns (output bytes, None) or (None, error)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agx-checked'))
    from run_checked import launch
    out = pack(c.out, c.out_init) if c.out_init is not None else bytes(N * TYPES[c.out][2])
    if c.scalars is not None:
        args = [('buf', out)] + [('param', [scalar_field(t, v)]) for t, v in c.scalars]
        index = 0
    else:
        args = [('buf', d) for d in inputs] + [('buf', out)]
        index = len(inputs)
    try:
        r = launch(obj, c.name, args, tuple(dispatch(c)[0]), tuple(dispatch(c)[1]), agx['run'], agx['objcopy'],
                   agx['readelf'])
    except (RuntimeError, KeyError, subprocess.TimeoutExpired) as e:
        return None, str(e)[-400:]
    return r.buffers[index], None


def run_group(cases, llc, queue, tmp, log, agx=None, gname='?', compare=False):
    """Results of the group's cases: on the device, on AGX (agx), or with compare on both, compared with each
    other (compare_backends)."""
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

    if compare:
        raw_air, raw_agx = {}, {}
        air_res = run_group_air(cases, data, llc, queue, tmp, log, gname, raw_air)
        agx_res = run_group_agx(cases, data, llc, tmp, agx, raw_agx)
        return compare_backends(cases, data, air_res, agx_res, raw_air, raw_agx)
    if agx:
        return run_group_agx(cases, data, llc, tmp, agx)
    return run_group_air(cases, data, llc, queue, tmp, log, gname)


def run_group_agx(cases, data, llc, tmp, agx, raw_out=None):
    results = {}
    for c in cases:
        if c.metal_only:
            results[c.name] = {'status': 'SKIPPED', 'detail': c.metal_only}
            continue
        obj, err = compile_agx(llc, module_ir([c]), tmp, c.name, agx['mcpu'])
        if obj is None:
            results[c.name] = {'status': 'ABSENT' if c.absent else 'COMPILE', 'detail': err}
            continue
        values = data[c.name]
        raw, err = run_case_agx(c, obj, [pack(t, v) for t, v in zip(c.ins, values)], agx)
        if raw is None:
            results[c.name] = {'status': 'REJECTED', 'detail': err}
            continue
        if raw_out is not None:
            raw_out[c.name] = raw
        results[c.name] = verdict(c, values, unpack(c.out, raw, N))
    return results


def run_group_air(cases, data, llc, queue, tmp, log, gname='?', raw_out=None):
    results = {}
    air, err = compile_air(llc, module_ir(cases), tmp, 'group')
    rejected_together = False
    if air is None:
        log('  group failed to compile, compiling cases alone')
    else:
        reply = call({'op': 'inspect', 'air': base64.b64encode(air).decode()}, queue, label='%s *' % gname)
        rejected_together = not reply.get('ok')

    for c in cases:
        mine = air
        if air is None or rejected_together:
            mine, err = compile_air(llc, module_ir([c]), tmp, c.name)
            if mine is None:
                results[c.name] = {'status': 'COMPILE', 'detail': err}
                continue
        values = data[c.name]
        reply = run_case(c, mine, [pack(t, v) for t, v in zip(c.ins, values)], queue, label='%s %s' % (gname, c.name))
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
        if raw_out is not None:
            raw_out[c.name] = raw
        results[c.name] = verdict(c, values, unpack(c.out, raw, N))
    return results


# --compare-backends: the class of a differing element, from the least to the most severe.
SEVERITY = ['ftz', 'nan', 'ulp', 'real']
FLOAT_BITS = {'f32': (32, 0x7f800000, 0x7fffff), 'f16': (16, 0x7c00, 0x3ff), 'bf16': (16, 0x7f80, 0x7f)}


def classify_diff(c, a, b, max_ulps=16):
    """(class, ULP distance or None) of two differing output encodings a (device) and b (AGX) of case c:
    ftz   one side is a float32 subnormal, the other a zero (one side flushed)
    nan   both NaNs (payload or sign only)
    ulp   a case with a tolerance (an approximate function) and within max(max_ulps, its ULP tolerance)
    real  anything else"""
    if c.out not in FLOAT_BITS:
        return 'real', None
    n, exp, man = FLOAT_BITS[c.out]
    nan = lambda x: (x & exp) == exp and x & man
    if nan(a) and nan(b):
        return 'nan', None
    if c.out == 'f32':
        ftz = lambda x: x & 0x80000000 if (x & exp) == 0 else x
        # one side is the other's value flushed (to a zero of either sign)
        if (ftz(a) != a and b & 0x7fffffff == 0) or (ftz(b) != b and a & 0x7fffffff == 0):
            return 'ftz', None
    if (c.ulps is not None or c.abstol is not None) and not nan(a) and not nan(b):
        sign = 1 << (n - 1)
        order = lambda x: -(x & (sign - 1)) if x & sign else x
        d = abs(order(a) - order(b))
        # an approximate function may differ by its own tolerance between two implementations
        if d <= max(max_ulps, c.ulps or 0):
            return 'ulp', d
    return 'real', None


def compare_backends(cases, data, air_res, agx_res, raw_air, raw_agx, max_diffs=4):
    """Each case's device output against its AGX output, bit for bit: SAME, or DIFF-<worst class> with the first
    differing threads; 'air' and 'agx' are each side's verdict against the reference."""
    results = {}
    for c in cases:
        ra, rg = air_res.get(c.name, {}), agx_res.get(c.name, {})
        r = {'air': ra.get('status'), 'agx': rg.get('status')}
        if c.name not in raw_air or c.name not in raw_agx:
            missing = [side for side, raw in (('air', raw_air), ('agx', raw_agx)) if c.name not in raw]
            r['status'] = 'NOT-COMPARED'
            r['detail'] = '; '.join('%s %s: %s' % (side, (ra if side == 'air' else rg).get('status'),
                                                   (ra if side == 'air' else rg).get('detail', '')[:200])
                                    for side in missing)
            results[c.name] = r
            continue
        size = TYPES[c.out][2]
        a, g = raw_air[c.name][:N * size], raw_agx[c.name][:N * size]
        classes, first, worst = {}, [], 0
        for i in range(N):
            x, y = int.from_bytes(a[i * size:(i + 1) * size], 'little'), int.from_bytes(g[i * size:(i + 1) * size],
                                                                                        'little')
            if x == y:
                continue
            cls, d = classify_diff(c, x, y)
            classes[cls] = classes.get(cls, 0) + 1
            if d is not None:
                worst = max(worst, d)
            if len(first) < max_diffs:
                ref = c.ref(i, data[c.name])
                first.append('thread %d: air=%#x agx=%#x %s%s (ref %r, inputs %r)' % (
                    i, x, y, cls, ' %d ulp' % d if d is not None else '', ref, [v[i] for v in data[c.name]]))
        if not classes:
            r['status'] = 'SAME'
        else:
            r['status'] = 'DIFF-' + max(classes, key=SEVERITY.index)
            r['classes'] = classes
            r['detail'] = '; '.join(first)
            if worst:
                r['max_ulps'] = worst
        results[c.name] = r
    return results


def is_sub32(x):
    """x is a float32 subnormal (nonzero)."""
    return isinstance(x, float) and x != 0 and abs(x) < FLT_MIN


def ftz32(x):
    return math.copysign(0.0, x) if is_sub32(x) else x


def hw_ftz(c, values, bad):
    """Whether every wrong result is explained by the hardware flushing
    float32 subnormals to zero: the thread has a subnormal float32 input or
    a subnormal exact result, and the device value, flushed, equals the
    reference computed from flushed inputs and flushed to float32."""
    f32_ins = [k for k, ty in enumerate(c.ins) if ty == 'f32']
    if not c.ftz_ok or not bad or (not f32_ins and c.out != 'f32'):
        return False
    flushed = [[ftz32(x) for x in vals] if ty == 'f32' else vals
               for ty, vals in zip(c.ins, values)]
    for i, g, w in bad:
        sub_in = any(is_sub32(values[k][i]) for k in f32_ins)
        sub_out = c.out == 'f32' and is_sub32(f32(w))
        if not (sub_in or sub_out):
            return False
        want = c.ref(i, flushed)
        if c.out == 'f32':
            g, want = ftz32(g), ftz32(f32(want))
        if g != g and want != want:
            continue
        if c.bitexact or c.out != 'f32':
            ok = float_bits(c.out, g) == float_bits(c.out, want) if c.out in ('f32', 'f16') else g == want
        elif c.ulps is not None:
            ok = ulp_distance(g, want) <= c.ulps or (c.abstol is not None and abs(g - want) <= c.abstol)
        else:
            ok = g == want or (c.abstol is not None and abs(g - want) <= c.abstol)
        if not ok:
            return False
    return True


def verdict(c, values, got):
    bad, worst = check(c, values, got)
    r = {'status': 'MISMATCH' if bad else 'PASS'}
    if bad and hw_ftz(c, values, bad):
        r['status'] = 'HW-FTZ'
    if c.ulps is not None:
        r['max_ulps'] = worst
    if bad:
        i, g, w = bad[0]
        r['detail'] = '%d/%d wrong; thread %d: got %r want %r (inputs %r)' % (
            len(bad), N, i, g, w, [v[i] for v in values])
    return r


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


def group_simd_divergent():
    """SIMD-group and quad-group operations inside divergent control flow: each
    sees only the threads of its SIMD-group that took the same path (Metal's
    compiler must not if-convert the branch; see AIRSimdDivergenceGuard)."""
    conds = {
        'lo': ('%lc = icmp ult i32 %lane, 16', lambda i: i % 32 < 16),
        'odd': ('%lo1 = and i32 %lane, 1\n%lc = icmp ne i32 %lo1, 0', lambda i: i % 2 == 1),
    }

    def active(i, cond):
        """Lanes (0..31) of thread i's SIMD-group on thread i's side of cond."""
        base = i // 32 * 32
        return [k for k in range(32) if cond(base + k) == cond(i)]

    def mask(lanes):
        return sum(1 << k for k in lanes)

    def diverge(name, ins, decls, a, b, cond, ref, gen=None):
        """if (cond) { a: defines %ra } else { b: defines %rb }; %r = phi."""
        pre, fn = conds[cond]
        body = (pre + '\nbr i1 %lc, label %sa, label %sb\n'
                'sa:\n' + a + '\nbr label %sj\n'
                'sb:\n' + b + '\nbr label %sj\n'
                'sj:\n%r = phi i32 [ %ra, %sa ], [ %rb, %sb ]')
        return ir_case(name, ins, 'u32', body, lambda i, v: ref(i, v, fn), decls, gen,
                       source='activemask-div.ptx (PTX-formalization)')

    ballot = 'declare i64 @air.simd_ballot.i64(i1)'
    lo = lambda i: i // 32 * 32
    cases = []

    def bt(v, p):
        return '%%m%s = call i64 @air.simd_ballot.i64(i1 %s)\n%%r%s = trunc i64 %%m%s to i32' % (v, p, v, v)
    cases.append(diverge('sdiv_ballot_true', ['u32'], [ballot], bt('a', 'true'), bt('b', 'true'), 'lo',
                         lambda i, v, c: mask(active(i, c))))
    pred = '%%od%s = trunc i32 %%a0 to i1\n'
    cases.append(diverge('sdiv_ballot_pred', ['u32'], [ballot], pred % 'a' + bt('a', '%oda'),
                         pred % 'b' + bt('b', '%odb'), 'odd',
                         lambda i, v, c: mask(k for k in active(i, c) if v[0][lo(i) + k] & 1)))
    cases.append(diverge('sdiv_ballot_one_sided', ['u32'], [ballot], bt('a', 'true'), '%rb = add i32 0, 0', 'lo',
                         lambda i, v, c: mask(active(i, c)) if c(i) else 0))
    cases.append(diverge('sdiv_active_threads_mask', ['u32'],
                         ['declare i64 @air.simd_active_threads_mask.i64()'],
                         '%ma = call i64 @air.simd_active_threads_mask.i64()\n%ra = trunc i64 %ma to i32',
                         '%mb = call i64 @air.simd_active_threads_mask.i64()\n%rb = trunc i64 %mb to i32', 'lo',
                         lambda i, v, c: mask(active(i, c))))

    def first(v):
        return '%%f%s = call i1 @air.simd_is_first()\n%%r%s = zext i1 %%f%s to i32' % (v, v, v)
    cases.append(diverge('sdiv_is_first', ['u32'], ['declare i1 @air.simd_is_first()'], first('a'), first('b'),
                         'lo', lambda i, v, c: int(i % 32 == active(i, c)[0])))
    cases.append(diverge('sdiv_broadcast_first', ['u32'], ['declare i32 @air.simd_broadcast_first.u.i32(i32)'],
                         '%ra = call i32 @air.simd_broadcast_first.u.i32(i32 %a0)',
                         '%rb = call i32 @air.simd_broadcast_first.u.i32(i32 %a0)', 'odd',
                         lambda i, v, c: v[0][lo(i) + active(i, c)[0]]))

    def vote(v, fn):
        return ('%%g%s = icmp ugt i32 %%a0, 3000000000\n%%m%s = call i64 @air.simd_ballot.i64(i1 %%g%s)\n'
                '%%v%s = call i1 @air.simd_vote_%s.i64(i64 %%m%s)\n%%r%s = zext i1 %%v%s to i32'
                % (v, v, v, v, fn, v, v, v))
    cases.append(diverge('sdiv_vote_any', ['u32'], [ballot, 'declare i1 @air.simd_vote_any.i64(i64)'],
                         vote('a', 'any'), vote('b', 'any'), 'lo',
                         lambda i, v, c: int(any(v[0][lo(i) + k] > 3000000000 for k in active(i, c))),
                         # only lane 20 votes: true on its side only
                         [lambda i, r: 4000000000 if i % 32 == 20 else r.randrange(0, 1 << 30)]))

    def red(v, fn, arg='%a0'):
        return '%%r%s = call i32 @air.%s(i32 %s)' % (v, fn, arg)
    cases.append(diverge('sdiv_sum', ['u32'], ['declare i32 @air.simd_sum.u.i32(i32)'],
                         red('a', 'simd_sum.u.i32'), red('b', 'simd_sum.u.i32'), 'odd',
                         lambda i, v, c: u32(sum(v[0][lo(i) + k] for k in active(i, c)))))
    cases.append(diverge('sdiv_prefix_exclusive_sum', ['u32'],
                         ['declare i32 @air.simd_prefix_exclusive_sum.u.i32(i32)'],
                         red('a', 'simd_prefix_exclusive_sum.u.i32'), red('b', 'simd_prefix_exclusive_sum.u.i32'),
                         'odd',
                         lambda i, v, c: u32(sum(v[0][lo(i) + k] for k in active(i, c) if k < i % 32))))
    cases.append(diverge('sdiv_min', ['u32'], ['declare i32 @air.simd_min.u.i32(i32)'],
                         red('a', 'simd_min.u.i32'), red('b', 'simd_min.u.i32'), 'lo',
                         lambda i, v, c: min(v[0][lo(i) + k] for k in active(i, c))))
    cases.append(diverge('sdiv_quad_sum', ['u32'], ['declare i32 @air.quad_sum.u.i32(i32)'],
                         red('a', 'quad_sum.u.i32'), red('b', 'quad_sum.u.i32'), 'odd',
                         lambda i, v, c: u32(sum(v[0][k] for k in range(i // 4 * 4, i // 4 * 4 + 4)
                                                 if c(k) == c(i)))))
    # A shuffle reads only lanes of its own side (xor 1 and xor 2 stay within a half).
    cases.append(diverge('sdiv_shuffle_xor', ['u32'], ['declare i32 @air.simd_shuffle_xor.u.i32(i32, i16)'],
                         '%ra = call i32 @air.simd_shuffle_xor.u.i32(i32 %a0, i16 1)',
                         '%rb = call i32 @air.simd_shuffle_xor.u.i32(i32 %a0, i16 2)', 'lo',
                         lambda i, v, c: v[0][i ^ 1] if c(i) else v[0][i ^ 2]))

    # A loop: thread i runs lane % 4 + 1 iterations; iteration k has the lanes with lane % 4 >= k active.
    def loop_ref(i, v):
        return u32(sum(mask(l for l in range(32) if l % 4 >= k) for k in range(i % 4 + 1)))
    cases.append(ir_case(
        'sdiv_loop_ballot', ['u32'], 'u32',
        '%n = and i32 %lane, 3\nbr label %lp\n'
        'lp:\n%k = phi i32 [ 0, %entry ], [ %k1, %lp ]\n%acc = phi i32 [ 0, %entry ], [ %acc1, %lp ]\n'
        '%m = call i64 @air.simd_ballot.i64(i1 true)\n%m32 = trunc i64 %m to i32\n'
        '%acc1 = add i32 %acc, %m32\n%k1 = add i32 %k, 1\n%more = icmp ule i32 %k1, %n\n'
        'br i1 %more, label %lp, label %lx\n'
        'lx:\n%r = add i32 %acc1, 0',
        loop_ref, [ballot], source='LLVM IR'))
    # After a divergent return: lane 31 of each SIMD-group has left.
    c = ir_case(
        'sdiv_after_return', ['u32'], 'u32',
        '%gone = icmp eq i32 %lane, 31\nbr i1 %gone, label %ex, label %go\n'
        'ex:\nret void\n'
        'go:\n%m = call i64 @air.simd_ballot.i64(i1 true)\n%r = trunc i64 %m to i32',
        lambda i, v: 0 if i % 32 == 31 else 0x7fffffff, [ballot], source='activemask-div.ptx (PTX-formalization)')
    c.out_init = [0] * N
    cases.append(c)
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
    c.metal_only = 'relies on air-tg-barrier-insert, which only the Metal pipeline runs'
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
        ('cvt_sitofp64', 'i64', 'f32', 'sitofp i64 %a to float', int_to_f32, None),
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


def group_extra():
    """The rest of the compute functions the reference front ends call."""
    cases = []
    # OpenCL-style builtin ids (dxmt): x component; threadgroups of 32.
    for name, ref in (('get_global_id', lambda i: i), ('get_local_id', lambda i: i % 32),
                      ('get_group_id', lambda i: i // 32)):
        c = Case('air_' + name, ['u32'], 'u32', '%%r = call i32 @air.%s.i32(i32 0)' % name,
                 {'declare i32 @air.%s.i32(i32)' % name}, lambda i, v, ref=ref: ref(i),
                 tgsize=32, source='dxmt CreateGetBuiltinID')
        cases.append(c)
    # air.convert, signed int to float.
    cases.append(Case('air_convert_f32_s32', ['i32'], 'f32',
                      '%r = call float @air.convert.f.f32.s.i32(i32 %a0)',
                      {'declare float @air.convert.f.f32.s.i32(i32)'},
                      lambda i, v: float(v[0][i]), ulps=0, source='dxmt CreateConvertToFloat'))
    # air.dot of two float4 (thread i < 16 handles 4 floats; others copy).
    cases.append(full_case(
        'air_dot_v4f32', ['f32', 'f32'], 'f32',
        lambda i, v: sum(v[0][j] * v[1][j] for j in range(i // 4 * 4, i // 4 * 4 + 4)) if i % 4 == 0 else 0.0,
        '%q = and i32 %tid, 15\n%is = icmp eq i32 %tid, %q\n'
        '%idx = shl i32 %tid, 2\n'
        'br i1 %is, label %go, label %done\n'
        'go:\n'
        '%pa = getelementptr <4 x float>, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = load <4 x float>, ptr addrspace(1) %pa\n'
        '%pb = getelementptr <4 x float>, ptr addrspace(1) %in1, i32 %tid\n'
        '%b = load <4 x float>, ptr addrspace(1) %pb\n'
        '%d = call float @air.dot.v4f32(<4 x float> %a, <4 x float> %b)\n'
        '%po = getelementptr float, ptr addrspace(1) %out, i32 %idx\n'
        'store float %d, ptr addrspace(1) %po\n'
        'br label %done\n'
        'done:\nret void',
        {'declare float @air.dot.v4f32(<4 x float>, <4 x float>)'},
        gen=[lambda i, r: float(r.randrange(-8, 9)), lambda i, r: float(r.randrange(-8, 9))],
        ulps=0, source='dxmt CreateDot'))
    cases[-1].out_init = [0.0] * N
    # pack/unpack unorm4x8: thread i < 16 packs in0[4i..4i+3]; out[i] holds the word.
    def pack_ref(i, v):
        if i >= 16:
            return 0
        b = [max(0, min(255, int(math.floor(max(0.0, min(1.0, x)) * 255 + 0.5))))
             for x in v[0][4 * i:4 * i + 4]]
        return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24
    c = full_case(
        'air_pack_unorm4x8', ['f32'], 'u32', pack_ref,
        '%small = icmp ult i32 %tid, 16\n'
        'br i1 %small, label %go, label %done\n'
        'go:\n'
        '%pa = getelementptr <4 x float>, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = load <4 x float>, ptr addrspace(1) %pa\n'
        '%w = call i32 @air.pack.unorm4x8.v4f32(<4 x float> %a)\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %w, ptr addrspace(1) %po\n'
        'br label %done\n'
        'done:\nret void',
        {'declare i32 @air.pack.unorm4x8.v4f32(<4 x float>)'},
        gen=[lambda i, r: r.uniform(-0.2, 1.2)], source='floor metal.hpp')
    c.out_init = [0] * N
    cases.append(c)
    # sincos: the cosine comes back through a private pointer.
    cases.append(full_case(
        'air_sincos', ['f32'], 'f32', lambda i, v: math.sin(v[0][i]) + 2 * math.cos(v[0][i]),
        '%cp = alloca float\n'
        '%pa = getelementptr float, ptr addrspace(1) %in0, i32 %tid\n'
        '%a = load float, ptr addrspace(1) %pa\n'
        '%sn = call float @air.sincos.f32(float %a, ptr %cp)\n'
        '%cs = load float, ptr %cp\n%c2 = fmul float %cs, 2.0\n%r = fadd float %sn, %c2\n'
        '%po = getelementptr float, ptr addrspace(1) %out, i32 %tid\n'
        'store float %r, ptr addrspace(1) %po\nret void',
        {'declare float @air.sincos.f32(float, ptr)'}, ulps=8, abstol=1e-6,
        source='Metal.jl sincos (pointer out-parameter)'))
    # shuffle_and_fill_down: lanes past the end take the fill value's lanes.
    cases.append(full_case(
        'simd_shuffle_and_fill_down', ['u32', 'u32'], 'u32',
        lambda i, v: lanes(i, v[0])[i % 32 + 3] if i % 32 + 3 < 32 else lanes(i, v[1])[(i % 32 + 3) - 32],
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
        '%pb = getelementptr i32, ptr addrspace(1) %in1, i32 %tid\n%b = load i32, ptr addrspace(1) %pb\n'
        '%r = call i32 @air.simd_shuffle_and_fill_down.u.i32(i32 %a, i32 %b, i16 3, i16 32)\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void',
        {'declare i32 @air.simd_shuffle_and_fill_down.u.i32(i32, i32, i16, i16)'},
        source='Metal.jl simd.jl'))
    # quad ballot (i16) and SIMD vote all.
    cases.append(full_case(
        'quad_ballot', ['u32'], 'u32',
        lambda i, v: sum((v[0][i // 4 * 4 + k] & 1) << k for k in range(4)),
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
        '%odd = trunc i32 %a to i1\n'
        '%m = call i16 @air.quad_ballot(i1 %odd)\n%m4 = and i16 %m, 15\n%r = zext i16 %m4 to i32\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void',
        {'declare i16 @air.quad_ballot(i1)'}, source='Metal.jl quad_ballot'))
    cases.append(full_case(
        'simd_vote_all', ['u32'], 'u32',
        lambda i, v: int(all(x < 4000000000 for x in lanes(i, v[0]))),
        '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
        '%small = icmp ult i32 %a, 4000000000\n'
        '%m = call i64 @air.simd_ballot.i64(i1 %small)\n'
        '%b = call i1 @air.simd_vote_all.i64(i64 %m)\n%r = zext i1 %b to i32\n'
        '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
        'store i32 %r, ptr addrspace(1) %po\nret void',
        {'declare i64 @air.simd_ballot.i64(i1)', 'declare i1 @air.simd_vote_all.i64(i64)'},
        source='Metal.jl simd_vote_all'))
    # simdgroup barrier on threadgroup memory within one SIMD-group.
    c = Case('simdgroup_barrier', ['i32'], 'i32', None,
             {'declare { i32, i32, i32 } @air.thread_position_in_threadgroup.v3i32()',
              'declare void @air.simdgroup.barrier(i32, i32)'},
             lambda i, v: v[0][i // 32 * 32 + 31 - i % 32], tgsize=32, source='dxmt CreateBarrier(SimdGroup)')
    c.full = tg_kernel('simdgroup_barrier', (
        '%sp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %lid\n'
        'store i32 %a0, ptr addrspace(3) %sp\n'
        'call void @air.simdgroup.barrier(i32 2, i32 4)\n'
        '%rl = sub i32 31, %lid\n'
        '%rp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %rl\n'
        '%r = load i32, ptr addrspace(3) %rp\n'
        'store i32 %r, ptr addrspace(1) %po'))
    cases.append(c)
    return cases


# ---------------------------------------------------------------------------
# LLVM IR semantics: special values, the integer and vector instructions,
# control flow, conversions and memory intrinsics, as a front end writes them
# ---------------------------------------------------------------------------

FLT_MAX = f32(3.4028234663852886e38)
FLT_MIN = 2.0 ** -126
DENORM = 2.0 ** -140  # a float32 subnormal
# Eight float32 values; their 64 ordered pairs are the 64 threads of a binary case.
F32_EDGE = [0.0, -0.0, math.inf, -math.inf, math.nan, DENORM, 1.0, -3.0]
# Values for unary cases, repeated over the threads.
F32_EDGE1 = F32_EDGE + [-DENORM, FLT_MIN, -FLT_MIN, FLT_MAX, -FLT_MAX, 0.5, -0.5, 1.5, 2.5, -2.5,
                        2.0 ** -149, 1e-30, 7.0, 8388609.0, -2.0 ** 31, 2.0 ** 32, 1e30, 0.75]
F32_FINITE = [x for x in F32_EDGE1 if math.isfinite(x)]
# Eight int32 values; their 64 ordered pairs are the threads of a compare case.
I32_EDGE = [0, 1, -1, -(1 << 31), (1 << 31) - 1, 2, -2, 0x10000]


def edge_a(i, rng):
    return F32_EDGE[i % 8]


def edge_b(i, rng):
    return F32_EDGE[i // 8]


def edge1(i, rng):
    return F32_EDGE1[i % len(F32_EDGE1)]


def finite1(i, rng):
    return F32_FINITE[i % len(F32_FINITE)]


def mag(i, rng):
    """Signed 32-bit values of every magnitude."""
    k = rng.randrange(1, 32)
    return rng.randrange(-(1 << k), 1 << k)


def umag(i, rng):
    return rng.randrange(0, 1 << rng.randrange(1, 33))


def umag64(i, rng):
    return rng.randrange(0, 1 << rng.randrange(1, 65))


def u64(x):
    return x % (1 << 64)


def tdiv(a, b):
    """C's (and LLVM's) division, truncating toward zero, and its remainder."""
    q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
    return q, a - b * q


def ir_case(name, ins, out, body, ref, decls=(), gen=None, ulps=None, bitexact=False,
            source='LLVM IR'):
    """A Case over the template kernel: body defines %r from %a0, %a1, ..."""
    c = Case(name, ins, out, body, set(decls), ref, gen, ulps, source)
    c.bitexact = bitexact
    return c


def ieee_div(a, b):
    if math.isnan(a) or math.isnan(b) or (a == 0 and b == 0) or (math.isinf(a) and math.isinf(b)):
        return math.nan
    if b == 0:
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def minnum(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return min(a, b)


def maxnum(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)


def minimum(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == b == 0:
        return -0.0 if math.copysign(1, a) < 0 or math.copysign(1, b) < 0 else 0.0
    return min(a, b)


def maximum(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if a == b == 0:
        return 0.0 if math.copysign(1, a) > 0 or math.copysign(1, b) > 0 else -0.0
    return max(a, b)


def fround(fn):
    """A rounding function with IEEE special cases: NaN and infinities pass,
    a zero result has the sign of the operand."""
    def r(x):
        if math.isnan(x) or math.isinf(x):
            return x
        y = float(fn(x))
        return math.copysign(y, x) if y == 0 else y
    return r


def fpclass(x):
    """The llvm.is.fpclass bit of x (inputs are quiet NaNs)."""
    neg = math.copysign(1, x) < 0
    if math.isnan(x):
        return 1 << 1
    if math.isinf(x):
        return 1 << (2 if neg else 9)
    if x == 0:
        return 1 << (5 if neg else 6)
    if abs(x) < FLT_MIN:
        return 1 << (4 if neg else 7)
    return 1 << (3 if neg else 8)


FCMP = {
    'oeq': lambda a, b: a == b, 'ogt': lambda a, b: a > b, 'oge': lambda a, b: a >= b,
    'olt': lambda a, b: a < b, 'ole': lambda a, b: a <= b,
    'one': lambda a, b: a == a and b == b and a != b,
    'ord': lambda a, b: a == a and b == b,
}
for _p, _f in list(FCMP.items()):
    if _p not in ('one', 'ord'):
        FCMP['u' + _p[1:]] = lambda a, b, f=_f: a != a or b != b or f(a, b)
FCMP['une'] = lambda a, b: a != b
FCMP['uno'] = lambda a, b: a != a or b != b


def group_llvm_float_ieee():
    """IEEE semantics of the float instructions on NaN, infinities, signed
    zeros and subnormals, which LLVM IR without fast-math flags requires."""
    cases = []
    edge2 = [edge_a, edge_b]
    for op, ref in (('fadd', lambda a, b: a + b), ('fsub', lambda a, b: a - b),
                    ('fmul', lambda a, b: a * b), ('fdiv', ieee_div),
                    ('frem', safe(math.fmod))):
        cases.append(ir_case('ieee_' + op, ['f32', 'f32'], 'f32', '%%r = %s float %%a0, %%a1' % op,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]), gen=edge2, ulps=0,
                             bitexact=True))
    # Correct rounding of the plain instructions (no afn/arcp: 0 ULPs).
    wide = lambda i, r: r.uniform(-1, 1) * 10.0 ** r.randrange(-6, 7)
    for op, ref in (('fdiv', ieee_div), ('frem', safe(math.fmod))):
        cases.append(ir_case('rand_' + op, ['f32', 'f32'], 'f32', '%%r = %s float %%a0, %%a1' % op,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]), gen=[wide, wide], ulps=0))
    # Subnormal operands and results (LLVM's default denormal mode is IEEE).
    sub = lambda i, r: r.uniform(-1, 1) * 2.0 ** -127
    tiny = lambda i, r: r.uniform(1, 2) * 2.0 ** -70
    cases.append(ir_case('denorm_fadd', ['f32', 'f32'], 'f32', '%r = fadd float %a0, %a1',
                         lambda i, v: v[0][i] + v[1][i], gen=[sub, sub], ulps=0, bitexact=True))
    cases.append(ir_case('denorm_fmul_result', ['f32', 'f32'], 'f32', '%r = fmul float %a0, %a1',
                         lambda i, v: v[0][i] * v[1][i], gen=[tiny, tiny], ulps=0, bitexact=True))
    cases.append(ir_case('denorm_fdiv_result', ['f32', 'f32'], 'f32', '%r = fdiv float %a0, %a1',
                         lambda i, v: v[0][i] / v[1][i], gen=[tiny, lambda i, r: r.uniform(1, 2) * 2.0 ** 70],
                         ulps=0, bitexact=True))
    cases.append(ir_case('ieee_fneg', ['f32'], 'f32', '%r = fneg float %a0', lambda i, v: -v[0][i],
                         gen=[edge1], ulps=0, bitexact=True))
    cases.append(ir_case('ieee_fsub_zero', ['f32'], 'f32', '%r = fsub float 0.0, %a0',
                         lambda i, v: 0.0 - v[0][i], gen=[edge1], ulps=0, bitexact=True,
                         source='not fneg: 0 - (+0) is +0'))
    for p, ref in sorted(FCMP.items()):
        cases.append(ir_case('fcmp_' + p, ['f32', 'f32'], 'u32',
                             '%%c = fcmp %s float %%a0, %%a1\n%%r = zext i1 %%c to i32' % p,
                             lambda i, v, ref=ref: int(ref(v[0][i], v[1][i])), gen=edge2))
    for name, ref, bitexact in (('minnum', minnum, False), ('maxnum', maxnum, False),
                                ('minimum', minimum, True), ('maximum', maximum, True),
                                ('copysign', math.copysign, True)):
        cases.append(ir_case('edge_' + name, ['f32', 'f32'], 'f32',
                             '%%r = call float @llvm.%s.f32(float %%a0, float %%a1)' % name,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]),
                             {'declare float @llvm.%s.f32(float, float)' % name}, edge2, 0, bitexact))
    rounding = (('fabs', abs), ('floor', math.floor), ('ceil', math.ceil), ('trunc', math.trunc),
                ('round', lambda x: math.copysign(math.floor(abs(x) + 0.5), x)),
                ('roundeven', round), ('rint', round), ('nearbyint', round),
                ('sqrt', lambda x: math.nan if x < 0 else math.sqrt(x)))
    for name, ref in rounding:
        r = (lambda x: x if math.isnan(x) else abs(x)) if name == 'fabs' else \
            (lambda x, ref=ref: x if x == 0 or math.isnan(x) or x == math.inf else ref(x)) \
            if name == 'sqrt' else fround(ref)
        cases.append(ir_case('edge_' + name, ['f32'], 'f32',
                             '%%r = call float @llvm.%s.f32(float %%a0)' % name,
                             lambda i, v, r=r: r(v[0][i]), {'declare float @llvm.%s.f32(float)' % name},
                             [edge1], 0, True))
    cases.append(ir_case('edge_ldexp', ['f32', 'i32'], 'f32',
                         '%r = call float @llvm.ldexp.f32.i32(float %a0, i32 %a1)',
                         lambda i, v: safe(math.ldexp)(v[0][i], v[1][i]) if math.isfinite(v[0][i])
                         else v[0][i], {'declare float @llvm.ldexp.f32.i32(float, i32)'},
                         [edge1, lambda i, r: r.randrange(-170, 170)], 0, True))
    cases.append(ir_case('frexp_exponent', ['f32'], 'i32',
                         '%fr = call { float, i32 } @llvm.frexp.f32.i32(float %a0)\n'
                         '%r = extractvalue { float, i32 } %fr, 1',
                         lambda i, v: math.frexp(v[0][i])[1],
                         {'declare { float, i32 } @llvm.frexp.f32.i32(float)'}, [finite1]))
    cases.append(ir_case('frexp_mantissa', ['f32'], 'f32',
                         '%fr = call { float, i32 } @llvm.frexp.f32.i32(float %a0)\n'
                         '%r = extractvalue { float, i32 } %fr, 0',
                         lambda i, v: math.frexp(v[0][i])[0],
                         {'declare { float, i32 } @llvm.frexp.f32.i32(float)'}, [edge1], 0, True))
    for mask in (0x3, 0x204, 0x60, 0x90, 0x108, 0x1c):
        cases.append(ir_case('is_fpclass_%#x' % mask, ['f32'], 'u32',
                             '%%c = call i1 @llvm.is.fpclass.f32(float %%a0, i32 %d)\n'
                             '%%r = zext i1 %%c to i32' % mask,
                             lambda i, v, m=mask: int(bool(fpclass(v[0][i]) & m)),
                             {'declare i1 @llvm.is.fpclass.f32(float, i32 immarg)'}, [edge1]))
    cases.append(ir_case('fmuladd', ['f32', 'f32', 'f32'], 'f32',
                         '%r = call float @llvm.fmuladd.f32(float %a0, float %a1, float %a2)',
                         lambda i, v: v[0][i] * v[1][i] + v[2][i],
                         {'declare float @llvm.fmuladd.f32(float, float, float)'}, ulps=1))
    return cases


def group_llvm_int_ops():
    """Shifts, 64-bit arithmetic, division by constants, saturating and
    overflow intrinsics, funnel shifts, byte swaps, integer compares."""
    cases = []
    sh32 = lambda i, r: r.randrange(0, 32)
    sh64 = lambda i, r: r.randrange(0, 64)
    for name, ty, inst, gen, ref in (
            ('shl_i32', 'u32', 'shl i32', sh32, lambda a, b: u32(a << b)),
            ('lshr_i32', 'u32', 'lshr i32', sh32, lambda a, b: a >> b),
            ('ashr_i32', 'i32', 'ashr i32', sh32, lambda a, b: a >> b),
            ('shl_i64', 'u64', 'shl i64', sh64, lambda a, b: u64(a << b)),
            ('lshr_i64', 'u64', 'lshr i64', sh64, lambda a, b: a >> b),
            ('ashr_i64', 'i64', 'ashr i64', sh64, lambda a, b: a >> b),
            ('mul_i64', 'u64', 'mul i64', None, lambda a, b: u64(a * b)),
            ('sdiv_i64', 'i64', 'sdiv i64', mag, lambda a, b: s64(tdiv(a, b)[0])),
            ('srem_i64', 'i64', 'srem i64', mag, lambda a, b: tdiv(a, b)[1]),
            ('urem_i64', 'u64', 'urem i64', umag64, lambda a, b: a % b),
            ('xor_i64', 'u64', 'xor i64', None, lambda a, b: a ^ b)):
        g2 = gen if gen else None
        if name.startswith(('sdiv', 'srem', 'urem')):
            g2 = lambda i, r, g=gen: g(i, r) or 1
        cases.append(ir_case(name, [ty, ty], ty, '%%r = %s %%a0, %%a1' % inst,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]), gen=[None, g2]))
    # Division by constants (strength-reduced to multiplies by the optimizer
    # or the backend).
    for name, ty, inst, ref in (
            ('udiv_c7', 'u32', 'udiv i32 %a0, 7', lambda a: a // 7),
            ('sdiv_cm3', 'i32', 'sdiv i32 %a0, -3', lambda a: tdiv(a, -3)[0]),
            ('urem_c10', 'u32', 'urem i32 %a0, 10', lambda a: a % 10),
            ('srem_c6', 'i32', 'srem i32 %a0, 6', lambda a: tdiv(a, 6)[1]),
            ('udiv_i64_c1000', 'u64', 'udiv i64 %a0, 1000', lambda a: a // 1000),
            ('sdiv_i32_pow2', 'i32', 'sdiv i32 %a0, 16', lambda a: tdiv(a, 16)[0])):
        cases.append(ir_case(name, [ty], ty, '%r = ' + inst, lambda i, v, ref=ref: ref(v[0][i])))
    clamp = lambda x, lo, hi: max(lo, min(hi, x))
    for name, intr, ty, ref in (
            ('sadd_sat_i32', 'llvm.sadd.sat.i32', 'i32', lambda a, b: clamp(a + b, -(1 << 31), (1 << 31) - 1)),
            ('ssub_sat_i32', 'llvm.ssub.sat.i32', 'i32', lambda a, b: clamp(a - b, -(1 << 31), (1 << 31) - 1)),
            ('uadd_sat_i32', 'llvm.uadd.sat.i32', 'u32', lambda a, b: min(a + b, (1 << 32) - 1)),
            ('usub_sat_i32', 'llvm.usub.sat.i32', 'u32', lambda a, b: max(a - b, 0)),
            ('uadd_sat_i8', 'llvm.uadd.sat.i8', 'u8', lambda a, b: min(a + b, 255)),
            ('sadd_sat_i16', 'llvm.sadd.sat.i16', 'i16', lambda a, b: clamp(a + b, -32768, 32767)),
            ('smin_i64', 'llvm.smin.i64', 'i64', min), ('umax_i64', 'llvm.umax.i64', 'u64', max)):
        cases.append(int_case(name, intr, ty, 2, lambda i, v, ref=ref: ref(v[0][i], v[1][i])))
    cases.append(int_case('abs_i64', 'llvm.abs.i64', 'i64', 1, lambda i, v: s64(abs(v[0][i])), ', i1 false'))
    for op, ty, gen, ref in (
            ('sadd', 'i32', mag, lambda a, b: (a + b, not -(1 << 31) <= a + b < 1 << 31)),
            ('uadd', 'u32', umag, lambda a, b: (a + b, a + b >= 1 << 32)),
            ('ssub', 'i32', mag, lambda a, b: (a - b, not -(1 << 31) <= a - b < 1 << 31)),
            ('usub', 'u32', umag, lambda a, b: (a - b, a < b)),
            ('smul', 'i32', mag, lambda a, b: (a * b, not -(1 << 31) <= a * b < 1 << 31)),
            ('umul', 'u32', umag, lambda a, b: (a * b, a * b >= 1 << 32))):
        def ref2(i, v, ref=ref):
            val, ovf = ref(v[0][i], v[1][i])
            return u32(val) << 1 | int(ovf)
        cases.append(ir_case(
            op + '_with_overflow', [ty, ty], 'u64',
            '%%s = call { i32, i1 } @llvm.%s.with.overflow.i32(i32 %%a0, i32 %%a1)\n'
            '%%v = extractvalue { i32, i1 } %%s, 0\n%%o = extractvalue { i32, i1 } %%s, 1\n'
            '%%v64 = zext i32 %%v to i64\n%%sh = shl i64 %%v64, 1\n%%o64 = zext i1 %%o to i64\n'
            '%%r = or i64 %%sh, %%o64' % op, ref2,
            {'declare { i32, i1 } @llvm.%s.with.overflow.i32(i32, i32)' % op}, [gen, gen]))
    M32 = (1 << 32) - 1
    for name, ref in (('fshl', lambda a, b, c: ((a << 32 | b) << (c % 32) >> 32) & M32),
                      ('fshr', lambda a, b, c: ((a << 32 | b) >> (c % 32)) & M32)):
        cases.append(int_case(name + '_i32', 'llvm.%s.i32' % name, 'u32', 3,
                              lambda i, v, ref=ref: ref(v[0][i], v[1][i], v[2][i])))
    for ty, n in (('u16', 2), ('u32', 4), ('u64', 8)):
        irty = TYPES[ty][0]
        cases.append(int_case('bswap_' + irty, 'llvm.bswap.' + irty, ty, 1,
                              lambda i, v, n=n: int.from_bytes(v[0][i].to_bytes(n, 'little'), 'big')))
    cases.append(int_case('ctpop_i64', 'llvm.ctpop.i64', 'u64', 1, lambda i, v: bin(v[0][i]).count('1')))
    cases.append(int_case('ctlz_i64', 'llvm.ctlz.i64', 'u64', 1, lambda i, v: 64 - v[0][i].bit_length(),
                          ', i1 false', umag64))
    cases.append(int_case('cttz_i64', 'llvm.cttz.i64', 'u64', 1,
                          lambda i, v: 64 if v[0][i] == 0 else (v[0][i] & -v[0][i]).bit_length() - 1,
                          ', i1 false', umag64))
    ia = lambda i, r: I32_EDGE[i % 8]
    ib = lambda i, r: I32_EDGE[i // 8]
    for p, ref in (('eq', lambda a, b: a == b), ('ne', lambda a, b: a != b),
                   ('sgt', lambda a, b: a > b), ('sge', lambda a, b: a >= b),
                   ('slt', lambda a, b: a < b), ('sle', lambda a, b: a <= b),
                   ('ugt', lambda a, b: u32(a) > u32(b)), ('uge', lambda a, b: u32(a) >= u32(b)),
                   ('ult', lambda a, b: u32(a) < u32(b)), ('ule', lambda a, b: u32(a) <= u32(b))):
        cases.append(ir_case('icmp_' + p, ['i32', 'i32'], 'u32',
                             '%%c = icmp %s i32 %%a0, %%a1\n%%r = zext i1 %%c to i32' % p,
                             lambda i, v, ref=ref: int(ref(v[0][i], v[1][i])), gen=[ia, ib]))
    for p, ref in (('slt', lambda a, b: a < b), ('ult', lambda a, b: u64(a) < u64(b))):
        cases.append(ir_case('icmp_i64_' + p, ['i64', 'i64'], 'u32',
                             '%%c = icmp %s i64 %%a0, %%a1\n%%r = zext i1 %%c to i32' % p,
                             lambda i, v, ref=ref: int(ref(v[0][i], v[1][i]))))
    cases.append(ir_case('freeze_add', ['i32'], 'i32', '%f = freeze i32 %a0\n%r = add i32 %f, 5',
                         lambda i, v: s32(v[0][i] + 5)))
    cases.append(ir_case('select_i64', ['i64', 'i64'], 'i64',
                         '%c = icmp sgt i64 %a0, %a1\n%r = select i1 %c, i64 %a1, i64 %a0',
                         lambda i, v: min(v[0][i], v[1][i])))
    return cases


VEC4_I32 = ('%v0 = insertelement <4 x i32> poison, i32 %a0, i32 0\n'
            '%v1 = insertelement <4 x i32> %v0, i32 %a1, i32 1\n'
            '%v2 = insertelement <4 x i32> %v1, i32 %a2, i32 2\n'
            '%v = insertelement <4 x i32> %v2, i32 %a3, i32 3\n'
            '%k = and i32 %tid, 3\n')
VEC4_F32 = VEC4_I32.replace('<4 x i32>', '<4 x float>').replace('i32 %a', 'float %a')


def group_llvm_vector():
    """insert/extractelement with dynamic indices, shufflevector, vector
    compares and selects, reductions, bitcasts between vector types."""
    U4 = ['u32'] * 4
    el = lambda v, k, i: v[k][i]
    cases = [
        ir_case('vec_extract_dynamic', U4, 'u32', VEC4_I32 + '%r = extractelement <4 x i32> %v, i32 %k',
                lambda i, v: el(v, i % 4, i)),
        ir_case('vec_insert_dynamic', U4, 'u32', VEC4_I32 +
                '%j0 = lshr i32 %tid, 2\n%j = and i32 %j0, 3\n'
                '%w = insertelement <4 x i32> %v, i32 99, i32 %j\n'
                '%r = extractelement <4 x i32> %w, i32 %k',
                lambda i, v: 99 if i % 4 == (i >> 2) % 4 else el(v, i % 4, i)),
        ir_case('vec_shuffle_reverse', U4, 'u32', VEC4_I32 +
                '%w = shufflevector <4 x i32> %v, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>\n'
                '%r = extractelement <4 x i32> %w, i32 %k', lambda i, v: el(v, 3 - i % 4, i)),
        ir_case('vec_shuffle_two', U4, 'u32', VEC4_I32 +
                '%w = add <4 x i32> %v, <i32 1000, i32 1000, i32 1000, i32 1000>\n'
                '%x = shufflevector <4 x i32> %v, <4 x i32> %w, <4 x i32> <i32 0, i32 5, i32 2, i32 7>\n'
                '%r = extractelement <4 x i32> %x, i32 %k',
                lambda i, v: u32(el(v, i % 4, i) + (1000 if i % 2 else 0))),
        ir_case('vec_icmp_select', U4, 'u32', VEC4_I32 +
                '%m = icmp ult <4 x i32> %v, <i32 2147483648, i32 2147483648, i32 2147483648, i32 2147483648>\n'
                '%n = xor <4 x i32> %v, <i32 -1, i32 -1, i32 -1, i32 -1>\n'
                '%s = select <4 x i1> %m, <4 x i32> %v, <4 x i32> %n\n'
                '%r = extractelement <4 x i32> %s, i32 %k',
                lambda i, v: el(v, i % 4, i) if el(v, i % 4, i) < 1 << 31 else el(v, i % 4, i) ^ 0xffffffff),
        ir_case('vec_bitcast_v8i16_mul', U4, 'u32', VEC4_I32 +
                '%h = bitcast <4 x i32> %v to <8 x i16>\n%h2 = mul <8 x i16> %h, %h\n'
                '%w = bitcast <8 x i16> %h2 to <4 x i32>\n%r = extractelement <4 x i32> %w, i32 %k',
                lambda i, v: ((el(v, i % 4, i) & 0xffff) ** 2 & 0xffff) |
                ((el(v, i % 4, i) >> 16) ** 2 & 0xffff) << 16),
    ]
    for op, ref in (('add', lambda xs: u32(sum(xs))), ('umax', max), ('umin', min),
                    ('xor', lambda xs: xs[0] ^ xs[1] ^ xs[2] ^ xs[3]),
                    ('and', lambda xs: xs[0] & xs[1] & xs[2] & xs[3]),
                    ('mul', lambda xs: u32(xs[0] * xs[1] * xs[2] * xs[3]))):
        cases.append(ir_case('vec_reduce_' + op, U4, 'u32',
                             VEC4_I32 + '%%r = call i32 @llvm.vector.reduce.%s.v4i32(<4 x i32> %%v)' % op,
                             lambda i, v, ref=ref: ref([v[k][i] for k in range(4)]),
                             {'declare i32 @llvm.vector.reduce.%s.v4i32(<4 x i32>)' % op}))
    cases.append(ir_case('vec_reduce_smax', ['i32'] * 4, 'i32',
                         VEC4_I32 + '%r = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %v)',
                         lambda i, v: max(v[k][i] for k in range(4)),
                         {'declare i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>)'}))
    spread = lambda i, r: r.uniform(-1, 1) * 10.0 ** r.randrange(-3, 8)

    def ordered_sum(i, v):
        acc = -0.0
        for k in range(4):
            acc = f32(acc + v[k][i])
        return acc
    cases.append(ir_case('vec_reduce_fadd_ordered', ['f32'] * 4, 'f32',
                         VEC4_F32 + '%r = call float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %v)',
                         ordered_sum, {'declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)'},
                         [spread] * 4, 0, True, source='strict (ordered) reduction'))
    cases.append(ir_case('vec_reduce_fmax', ['f32'] * 4, 'f32',
                         VEC4_F32 + '%r = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %v)',
                         lambda i, v: max(v[k][i] for k in range(4)),
                         {'declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>)'}, [spread] * 4, 0))
    cases.append(ir_case('vec_v2i64_add', ['u64', 'u64'], 'u64',
                         '%x0 = insertelement <2 x i64> poison, i64 %a0, i32 0\n'
                         '%x = insertelement <2 x i64> %x0, i64 %a1, i32 1\n'
                         '%y = shufflevector <2 x i64> %x, <2 x i64> poison, <2 x i32> <i32 1, i32 0>\n'
                         '%z = add <2 x i64> %x, %y\n%k = and i32 %tid, 1\n'
                         '%r = extractelement <2 x i64> %z, i32 %k',
                         lambda i, v: u64(v[0][i] + v[1][i])))
    cases.append(ir_case('vec_v4f16_fma', ['f16'] * 4, 'f16',
                         VEC4_F32.replace('float', 'half') +
                         '%w = call <4 x half> @llvm.fma.v4f16(<4 x half> %v, <4 x half> %v, <4 x half> %v)\n'
                         '%r = extractelement <4 x half> %w, i32 %k',
                         lambda i, v: f16(v[i % 4][i] * v[i % 4][i] + v[i % 4][i]),
                         {'declare <4 x half> @llvm.fma.v4f16(<4 x half>, <4 x half>, <4 x half>)'},
                         [lambda i, r: r.uniform(-8, 8)] * 4, 0, True))
    return cases


CF_HELPER = '''define internal i32 @cf_helper(i32 %x, i32 %y) noinline {
entry:
  %c = icmp ugt i32 %x, %y
  br i1 %c, label %a, label %b
a:
  %d = sub i32 %x, %y
  ret i32 %d
b:
  %m = shl i32 %y, 1
  %e = or i32 %m, 1
  ret i32 %e
}'''


def group_llvm_control():
    """switch, loops with data-dependent trip counts, break/continue,
    divergent returns, calls to non-kernel functions."""
    small10 = lambda i, r: r.randrange(0, 1 << 10)
    cases = []

    def switch_ref(i, v):
        a = v[0][i]
        return {0: 11, 1: u32(a * 3), 2: a >> 4, 5: 0x12345678}.get(a & 7, a ^ 85)
    cases.append(ir_case(
        'cf_switch', ['u32'], 'u32',
        '%sel = and i32 %a0, 7\n'
        'switch i32 %sel, label %sd [ i32 0, label %s0 i32 1, label %s1 i32 2, label %s2 i32 5, label %s5 ]\n'
        's0:\nbr label %sj\n'
        's1:\n%v1 = mul i32 %a0, 3\nbr label %sj\n'
        's2:\n%v2 = lshr i32 %a0, 4\nbr label %sj\n'
        's5:\nbr label %sj\n'
        'sd:\n%vd = xor i32 %a0, 85\nbr label %sj\n'
        'sj:\n%r = phi i32 [ 11, %s0 ], [ %v1, %s1 ], [ %v2, %s2 ], [ 305419896, %s5 ], [ %vd, %sd ]',
        switch_ref))

    def collatz(n):
        c = 0
        while n != 1:
            n = 3 * n + 1 if n & 1 else n >> 1
            c += 1
        return c
    cases.append(ir_case(
        'cf_loop_collatz', ['u32'], 'u32',
        '%n0 = and i32 %a0, 1023\n%n1 = add i32 %n0, 1\nbr label %loop\n'
        'loop:\n%n = phi i32 [ %n1, %entry ], [ %next, %body ]\n'
        '%c = phi i32 [ 0, %entry ], [ %c1, %body ]\n'
        '%done = icmp eq i32 %n, 1\nbr i1 %done, label %exit, label %body\n'
        'body:\n%odd = and i32 %n, 1\n%isodd = icmp ne i32 %odd, 0\n%h = lshr i32 %n, 1\n'
        '%m3 = mul i32 %n, 3\n%t3 = add i32 %m3, 1\n%next = select i1 %isodd, i32 %t3, i32 %h\n'
        '%c1 = add i32 %c, 1\nbr label %loop\n'
        'exit:\n%r = add i32 %c, 0',
        lambda i, v: collatz((v[0][i] & 1023) + 1)))
    cases.append(ir_case(
        'cf_nested_loop', ['u32', 'u32'], 'u32',
        '%na = and i32 %a0, 15\n%nb = and i32 %a1, 15\nbr label %oh\n'
        'oh:\n%i = phi i32 [ 0, %entry ], [ %i1, %ol ]\n%acc = phi i32 [ 0, %entry ], [ %acc1, %ol ]\n'
        '%ci = icmp ult i32 %i, %na\nbr i1 %ci, label %ih, label %exit\n'
        'ih:\n%j = phi i32 [ 0, %oh ], [ %j1, %ib ]\n%acc1 = phi i32 [ %acc, %oh ], [ %accn, %ib ]\n'
        '%cj = icmp ult i32 %j, %nb\nbr i1 %cj, label %ib, label %ol\n'
        'ib:\n%ij = mul i32 %i, %j\n%x = xor i32 %ij, %tid\n%accn = add i32 %acc1, %x\n'
        '%j1 = add i32 %j, 1\nbr label %ih\n'
        'ol:\n%i1 = add i32 %i, 1\nbr label %oh\n'
        'exit:\n%r = add i32 %acc, 0',
        lambda i, v: u32(sum((a * b) ^ i for a in range(v[0][i] & 15) for b in range(v[1][i] & 15)))))
    cases.append(ir_case(
        'cf_break_continue', ['u32', 'u32'], 'u32',
        '%stop = and i32 %a1, 31\nbr label %h\n'
        'h:\n%k = phi i32 [ 0, %entry ], [ %k1, %latch ]\n%s = phi i32 [ 0, %entry ], [ %s2, %latch ]\n'
        '%atstop = icmp eq i32 %k, %stop\nbr i1 %atstop, label %exit, label %b\n'
        'b:\n%bit = lshr i32 %a0, %k\n%bit1 = and i32 %bit, 1\n%skip = icmp ne i32 %bit1, 0\n'
        'br i1 %skip, label %latch0, label %acc\n'
        'acc:\n%sk = mul i32 %k, %k\n%s1 = add i32 %s, %sk\nbr label %latch\n'
        'latch0:\nbr label %latch\n'
        'latch:\n%s2 = phi i32 [ %s1, %acc ], [ %s, %latch0 ]\n%k1 = add i32 %k, 1\nbr label %h\n'
        'exit:\n%r = add i32 %s, 0',
        lambda i, v: sum(k * k for k in range(v[1][i] & 31) if not v[0][i] >> k & 1)))
    cases.append(ir_case('cf_call_internal', ['u32', 'u32'], 'u32',
                         '%r = call i32 @cf_helper(i32 %a0, i32 %a1)',
                         lambda i, v: u32(v[0][i] - v[1][i]) if v[0][i] > v[1][i] else u32(v[1][i] * 2 + 1),
                         {CF_HELPER}, [small10, small10]))
    c = full_case('cf_divergent_return', ['u32'], 'u32',
                  lambda i, v: 0xdeadbeef if v[0][i] & 1 else u32(v[0][i] + 1),
                  '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
                  '%a = load i32, ptr addrspace(1) %pa\n%odd = and i32 %a, 1\n'
                  '%isodd = icmp ne i32 %odd, 0\nbr i1 %isodd, label %early, label %st\n'
                  'early:\nret void\n'
                  'st:\n%r = add i32 %a, 1\n'
                  '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                  'store i32 %r, ptr addrspace(1) %po\nret void', source='LLVM IR')
    c.out_init = [0xdeadbeef] * N
    cases.append(c)
    return cases


def bf16_bits_to_f32(b):
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]


def f32_to_bf16_bits(x):
    """float32 to bfloat16, to nearest even (x is not a NaN)."""
    u = struct.unpack('<I', struct.pack('<f', x))[0]
    return ((u + 0x7fff + (u >> 16 & 1)) >> 16) & 0xffff


def group_llvm_conv():
    """Saturating conversions, float <-> 64-bit integers, half and bfloat."""
    cases = []
    i32_range = (-(1 << 31), (1 << 31) - 1)
    u32_range = (0, (1 << 32) - 1)
    for name, ty, rng_ in (('fptosi_sat', 'i32', i32_range), ('fptoui_sat', 'u32', u32_range)):
        cases.append(ir_case(
            name + '_i32', ['f32'], ty, '%%r = call i32 @llvm.%s.i32.f32(float %%a0)' % name.replace('_', '.'),
            lambda i, v, lo_hi=rng_: 0 if math.isnan(v[0][i]) else
            max(lo_hi[0], min(lo_hi[1], int(v[0][i]) if math.isfinite(v[0][i]) else
                              (lo_hi[1] if v[0][i] > 0 else lo_hi[0]))),
            {'declare i32 @llvm.%s.i32.f32(float)' % name.replace('_', '.')}, [edge1]))
    cases.append(ir_case('fptosi_f32_i64', ['f32'], 'i64', '%r = fptosi float %a0 to i64',
                         lambda i, v: int(v[0][i]), gen=[lambda i, r: r.uniform(-9e18, 9e18)]))
    cases.append(ir_case('fptoui_f32_u64', ['f32'], 'u64', '%r = fptoui float %a0 to i64',
                         lambda i, v: int(v[0][i]), gen=[lambda i, r: r.uniform(0, 1.8e19)]))

    def ties64(i, r):
        """Mostly 64-bit integers exactly halfway between two float32 values."""
        if i % 4 == 3:
            return r.randrange(0, 1 << 64)
        return ((r.randrange(1 << 23, 1 << 24) << 1) | 1) << r.randrange(0, 39)
    cases.append(ir_case('uitofp_u64_f32', ['u64'], 'f32', '%r = uitofp i64 %a0 to float',
                         lambda i, v: int_to_f32(v[0][i]), gen=[ties64], ulps=0))
    cases.append(ir_case('sitofp_i64_f32_ties', ['i64'], 'f32', '%r = sitofp i64 %a0 to float',
                         lambda i, v: int_to_f32(v[0][i]),
                         gen=[lambda i, r: s64(ties64(i, r) >> 1) * (-1 if i & 1 else 1)], ulps=0))
    cases.append(ir_case('fptosi_f16_i32', ['f16'], 'i32', '%r = fptosi half %a0 to i32',
                         lambda i, v: int(v[0][i]), gen=[lambda i, r: r.uniform(-2000, 2000)]))
    c = ir_case('uitofp_u32_f16', ['u32'], 'f16', '%r = uitofp i32 %a0 to half',
                lambda i, v: f16(float(v[0][i])), gen=[lambda i, r: r.randrange(0, 70000)], ulps=0,
                bitexact=True)
    cases.append(c)
    half_edges = [65520.0, 65519.0, -65504.0, 2.0 ** -24, 2.0 ** -25, 1.5 * 2.0 ** -24, 2.0 ** -14,
                  1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, -0.0, math.inf, math.nan]
    cases.append(ir_case('fptrunc_f32_f16_edges', ['f32'], 'f16', '%r = fptrunc float %a0 to half',
                         lambda i, v: f16(v[0][i]),
                         gen=[lambda i, r: half_edges[i] if i < len(half_edges) else
                              r.choice([r.uniform(-70000, 70000), r.uniform(-1e-4, 1e-4),
                                        r.uniform(-1e-7, 1e-7)])], ulps=0, bitexact=True))
    cases.append(ir_case('fpext_bf16_f32', ['bf16'], 'f32', '%r = fpext bfloat %a0 to float',
                         lambda i, v: bf16_bits_to_f32(v[0][i]),
                         gen=[lambda i, r: r.randrange(0, 1 << 16)], ulps=0, bitexact=True))

    def bf16_src(i, r):
        sign = 0x80000000 if r.randrange(2) else 0
        low = r.choice([0x8000, 0x7fff, 0x8001, 0, r.randrange(0, 1 << 16)])
        return struct.unpack('<f', struct.pack('<I', sign | r.randrange(0, 0x7f80) << 16 | low))[0]
    cases.append(ir_case('fptrunc_f32_bf16', ['f32'], 'bf16', '%r = fptrunc float %a0 to bfloat',
                         lambda i, v: f32_to_bf16_bits(v[0][i]), gen=[bf16_src]))
    return cases


def group_llvm_memory_ir():
    """memset (constant and dynamic length), byte and halfword stores from
    neighbouring threads, unaligned and packed accesses, ptrtoint/inttoptr."""
    cases = [
        ir_case('memset_private', ['u32'], 'u32',
                '%arr = alloca [16 x i32], align 4\n%bv = trunc i32 %a0 to i8\n'
                'call void @llvm.memset.p0.i32(ptr align 4 %arr, i8 %bv, i32 64, i1 false)\n'
                '%k = and i32 %tid, 15\n%ep = getelementptr [16 x i32], ptr %arr, i32 0, i32 %k\n'
                '%r = load i32, ptr %ep', lambda i, v: (v[0][i] & 0xff) * 0x01010101,
                {'declare void @llvm.memset.p0.i32(ptr, i8, i32, i1 immarg)'}),
        ir_case('store_bytes', ['u32'], 'u8', '%m = mul i32 %a0, 3\n%r = trunc i32 %m to i8',
                lambda i, v: v[0][i] * 3 & 0xff, source='adjacent threads write one word'),
        ir_case('store_halfwords', ['u32'], 'u16', '%m = xor i32 %a0, %tid\n%r = trunc i32 %m to i16',
                lambda i, v: (v[0][i] ^ i) & 0xffff),
    ]
    c = full_case('memset_device_dynamic', ['u32', 'u32'], 'u32',
                  lambda i, v: (v[0][i] & 0xff) * (0x01010101 & ((1 << 8 * ((v[1][i] & 3) + 1)) - 1)),
                  '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
                  '%pb = getelementptr i32, ptr addrspace(1) %in1, i32 %tid\n%b = load i32, ptr addrspace(1) %pb\n'
                  '%bv = trunc i32 %a to i8\n%n0 = and i32 %b, 3\n%n = add i32 %n0, 1\n'
                  '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                  'call void @llvm.memset.p1.i32(ptr addrspace(1) align 4 %po, i8 %bv, i32 %n, i1 false)\n'
                  'ret void', {'declare void @llvm.memset.p1.i32(ptr addrspace(1), i8, i32, i1 immarg)'},
                  source='LLVM IR')
    c.out_init = [0] * N
    cases.append(c)
    le = lambda bs: int.from_bytes(bytes(bs), 'little')
    cases.append(full_case('load_unaligned_i32', ['u8'], 'u32', lambda i, v: le(v[0][i % 60:i % 60 + 4]),
                           '%off = urem i32 %tid, 60\n%pb = getelementptr i8, ptr addrspace(1) %in0, i32 %off\n'
                           '%r = load i32, ptr addrspace(1) %pb, align 1\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store i32 %r, ptr addrspace(1) %po\nret void', source='LLVM IR'))
    cases.append(full_case('load_unaligned_i64', ['u8'], 'u64', lambda i, v: le(v[0][i % 56:i % 56 + 8]),
                           '%off = urem i32 %tid, 56\n%pb = getelementptr i8, ptr addrspace(1) %in0, i32 %off\n'
                           '%r = load i64, ptr addrspace(1) %pb, align 1\n'
                           '%po = getelementptr i64, ptr addrspace(1) %out, i32 %tid\n'
                           'store i64 %r, ptr addrspace(1) %po\nret void', source='LLVM IR'))
    cases.append(full_case('load_packed_struct', ['u8'], 'u32',
                           lambda i, v: le(v[0][5 * (i % 12) + 1:5 * (i % 12) + 5]),
                           '%e = urem i32 %tid, 12\n'
                           '%pf = getelementptr <{ i8, i32 }>, ptr addrspace(1) %in0, i32 %e, i32 1\n'
                           '%r = load i32, ptr addrspace(1) %pf, align 1\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store i32 %r, ptr addrspace(1) %po\nret void', source='LLVM IR'))
    cases.append(full_case('ptrtoint_inttoptr', ['u32'], 'u32', lambda i, v: v[0][(i + 1) % N],
                           '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n'
                           '%ia = ptrtoint ptr addrspace(1) %pa to i64\n'
                           '%n = add i32 %tid, 1\n%nm = and i32 %n, 63\n%d = sub i32 %nm, %tid\n'
                           '%d64 = sext i32 %d to i64\n%d4 = mul i64 %d64, 4\n%ib = add i64 %ia, %d4\n'
                           '%pb = inttoptr i64 %ib to ptr addrspace(1)\n'
                           '%r = load i32, ptr addrspace(1) %pb\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store i32 %r, ptr addrspace(1) %po\nret void', source='LLVM IR'))
    return cases


# ---------------------------------------------------------------------------
# Front-end IR, continued: half and bfloat, narrow and wide integers, more
# vectors, math, control flow, memory, conversions, and 2D/3D dispatch
# ---------------------------------------------------------------------------

# (significand bits, exponent of the smallest normal, largest exponent)
FORMATS = {'f32': (24, -126, 127), 'f16': (11, -14, 15), 'bf16': (8, -126, 127)}


def round_to(ty, x):
    """The exact value x (a Fraction, int or finite float) rounded to nearest
    even in the format ty, subnormals and overflow included."""
    p, emin, emax = FORMATS[ty]
    x = Fraction(x)
    if x == 0:
        return 0.0
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    q = Fraction(2) ** (max(e, emin) - p + 1)
    r = round(a / q) * q
    res = math.inf if r >= Fraction(2) ** (emax + 1) else float(r)
    return -res if x < 0 else res


def fp_op(ty, op, a, b):
    """The IEEE result of a op b in the format ty (a, b values of ty)."""
    if op == 'fdiv':
        base = ieee_div(a, b)
    else:
        base = {'fadd': a + b, 'fsub': a - b, 'fmul': a * b}[op]
    # Special operands, and zeros (which doubles compute exactly here).
    if not (math.isfinite(a) and math.isfinite(b)) or not math.isfinite(base) or base == 0:
        return base
    fa, fb = Fraction(a), Fraction(b)
    ops = {'fadd': lambda: fa + fb, 'fsub': lambda: fa - fb, 'fmul': lambda: fa * fb,
           'fdiv': lambda: fa / fb}
    return round_to(ty, ops[op]())


def fma_ref(ty, a, b, c):
    """llvm.fma: a * b + c rounded once."""
    if not all(math.isfinite(x) for x in (a, b, c)):
        return a * b + c if not (math.isinf(a) and b == 0 or math.isinf(b) and a == 0) else math.nan
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        return (a * b) + c if (a == 0 or b == 0) and c == 0 else 0.0
    return round_to(ty, exact)


def cyc(vals, k=1):
    """A generator: thread i gets vals[(i // k) % len(vals)]."""
    return lambda i, r: vals[(i // k) % len(vals)]


def bf16_bits(x):
    """The bfloat16 encoding of x, which is a bfloat16 value or NaN."""
    return 0x7fc0 if x != x else f32_to_bf16_bits(x)


def f16_bits(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def bits_f16(b):
    return struct.unpack('<e', struct.pack('<H', b))[0]


F16_MIN = 2.0 ** -14
F16_SUB = 2.0 ** -20  # a half subnormal
F16_EDGE = [0.0, -0.0, math.inf, -math.inf, math.nan, F16_SUB, 1.0, -3.0]
F16_EDGE1 = F16_EDGE + [-F16_SUB, 2.0 ** -24, F16_MIN, -F16_MIN, 65504.0, -65504.0, 0.5, -0.5,
                        1.5, 2.5, -2.5, 0.75, 1023.5, 3.0 * 2.0 ** -16, 7.0, 2050.0, -0.1, 0.1]


def wide16(i, r):
    """Halves of every magnitude: products and quotients overflow and underflow."""
    return r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randrange(-14, 9)


def bf16_gen(lo=-20, hi=20):
    """Normal bfloat16 encodings with exponents in [lo, hi)."""
    return lambda i, r: (r.randrange(2) << 15) | ((r.randrange(lo, hi) + 127) << 7) | r.randrange(128)


def ieee_sqrt(ty):
    return lambda x: x if x != x or x == 0 or x == math.inf else (
        math.nan if x < 0 else round_to(ty, Fraction(math.sqrt(x))))


def group_llvm_half():
    """half and bfloat arithmetic on special values, half subnormals,
    conversions between half, bfloat and integers."""
    cases = []
    pair = [cyc(F16_EDGE), cyc(F16_EDGE, 8)]
    for op in ('fadd', 'fsub', 'fmul', 'fdiv'):
        cases.append(ir_case('half_' + op, ['f16', 'f16'], 'f16', '%%r = %s half %%a0, %%a1' % op,
                             lambda i, v, op=op: fp_op('f16', op, v[0][i], v[1][i]), gen=pair,
                             bitexact=True))
    for op in ('fadd', 'fmul', 'fdiv'):
        cases.append(ir_case('half_rand_' + op, ['f16', 'f16'], 'f16', '%%r = %s half %%a0, %%a1' % op,
                             lambda i, v, op=op: fp_op('f16', op, v[0][i], v[1][i]),
                             gen=[wide16, wide16], bitexact=True))
    # Half subnormals: operands, results, and their conversion to float.
    hsub = lambda i, r: r.randrange(-1023, 1024) * 2.0 ** -24
    hsm = lambda i, r: r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randrange(-9, -6)
    cases.append(ir_case('half_denorm_fadd', ['f16', 'f16'], 'f16', '%r = fadd half %a0, %a1',
                         lambda i, v: fp_op('f16', 'fadd', v[0][i], v[1][i]), gen=[hsub, hsub],
                         bitexact=True))
    cases.append(ir_case('half_denorm_fmul_result', ['f16', 'f16'], 'f16', '%r = fmul half %a0, %a1',
                         lambda i, v: fp_op('f16', 'fmul', v[0][i], v[1][i]), gen=[hsm, hsm],
                         bitexact=True))
    cases.append(ir_case('half_denorm_fpext', ['f16'], 'f32', '%r = fpext half %a0 to float',
                         lambda i, v: v[0][i], gen=[hsub], bitexact=True))
    cases.append(ir_case('half_denorm_fcmp_oeq_zero', ['f16'], 'u32',
                         '%c = fcmp oeq half %a0, 0.0\n%r = zext i1 %c to i32',
                         lambda i, v: int(v[0][i] == 0), gen=[hsub]))
    for p in ('oeq', 'one', 'olt', 'oge', 'ord', 'uno', 'ueq', 'ult', 'une'):
        cases.append(ir_case('half_fcmp_' + p, ['f16', 'f16'], 'u32',
                             '%%c = fcmp %s half %%a0, %%a1\n%%r = zext i1 %%c to i32' % p,
                             lambda i, v, f=FCMP[p]: int(f(v[0][i], v[1][i])), gen=pair))
    for name, ref, bitexact in (('minnum', minnum, False), ('maxnum', maxnum, False),
                                ('copysign', math.copysign, True)):
        cases.append(ir_case('half_' + name, ['f16', 'f16'], 'f16',
                             '%%r = call half @llvm.%s.f16(half %%a0, half %%a1)' % name,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]),
                             {'declare half @llvm.%s.f16(half, half)' % name}, pair, 0, bitexact))
    cases.append(ir_case('half_fneg', ['f16'], 'f16', '%r = fneg half %a0', lambda i, v: -v[0][i],
                         gen=[cyc(F16_EDGE1)], ulps=0, bitexact=True))
    for name, ref in (('fabs', lambda x: x if x != x else abs(x)), ('floor', fround(math.floor)),
                      ('ceil', fround(math.ceil)), ('trunc', fround(math.trunc)),
                      ('rint', fround(round)), ('sqrt', ieee_sqrt('f16'))):
        cases.append(ir_case('half_' + name, ['f16'], 'f16',
                             '%%r = call half @llvm.%s.f16(half %%a0)' % name,
                             lambda i, v, ref=ref: ref(v[0][i]), {'declare half @llvm.%s.f16(half)' % name},
                             [cyc(F16_EDGE1)], 0, True))
    fma_decl = {'declare half @llvm.fma.f16(half, half, half)'}
    cases.append(ir_case('half_fma_edges', ['f16'] * 3, 'f16',
                         '%r = call half @llvm.fma.f16(half %a0, half %a1, half %a2)',
                         lambda i, v: fma_ref('f16', v[0][i], v[1][i], v[2][i]), fma_decl,
                         pair + [lambda i, r: F16_EDGE1[i * 7 % len(F16_EDGE1)]], 0, True))
    cases.append(ir_case('half_fma_rand', ['f16'] * 3, 'f16',
                         '%r = call half @llvm.fma.f16(half %a0, half %a1, half %a2)',
                         lambda i, v: fma_ref('f16', v[0][i], v[1][i], v[2][i]), fma_decl,
                         [wide16] * 3, 0, True))
    # Integer conversions (in range: out of range is poison).
    for name, ti, to, inst, ref, gen in (
            ('fptosi_f16_i16', 'f16', 'i16', 'fptosi half %a0 to i16', int,
             lambda i, r: r.uniform(-32000, 32000)),
            ('fptoui_f16_u16', 'f16', 'u16', 'fptoui half %a0 to i16', int,
             lambda i, r: r.uniform(0, 65504)),
            ('fptosi_f16_i8', 'f16', 'i8', 'fptosi half %a0 to i8', int,
             lambda i, r: r.uniform(-128.9, 127.9)),
            ('fptosi_f16_i64', 'f16', 'i64', 'fptosi half %a0 to i64', int,
             lambda i, r: r.uniform(-65504, 65504)),
            ('fptoui_f16_u8', 'f16', 'u8', 'fptoui half %a0 to i8', int,
             lambda i, r: r.uniform(0, 255.9)),
            ('sitofp_i16_f16', 'i16', 'f16', 'sitofp i16 %a0 to half', lambda x: round_to('f16', x), None),
            ('uitofp_u16_f16', 'u16', 'f16', 'uitofp i16 %a0 to half', lambda x: round_to('f16', x), None),
            ('sitofp_i8_f16', 'i8', 'f16', 'sitofp i8 %a0 to half', float, None),
            ('sitofp_i32_f16', 'i32', 'f16', 'sitofp i32 %a0 to half', lambda x: round_to('f16', x),
             lambda i, r: r.randrange(-100000, 100000)),
            ('sitofp_i64_f16', 'i64', 'f16', 'sitofp i64 %a0 to half', lambda x: round_to('f16', x),
             lambda i, r: r.choice([r.randrange(-70000, 70000), r.randrange(-(1 << 63), 1 << 63)]))):
        c = ir_case(name, [ti], to, '%r = ' + inst, lambda i, v, ref=ref: ref(v[0][i]), gen=[gen],
                    bitexact=to == 'f16')
        cases.append(c)
    # The sign operations on encodings, NaNs included: they only touch the sign bit.
    any16 = lambda i, r: r.choice([r.randrange(0, 1 << 16), 0x7c01 + r.randrange(0x3ff),
                                   0xfe00 + r.randrange(0x200), r.randrange(1, 0x400)])
    for ty, irty in (('half', 'f16'), ('bfloat', 'bf16')):
        sign = 0x8000
        cases.append(ir_case('%s_fneg_bits' % irty, ['u16'], 'u16',
                             '%%b = bitcast i16 %%a0 to %s\n%%n = fneg %s %%b\n%%r = bitcast %s %%n to i16'
                             % (ty, ty, ty), lambda i, v: v[0][i] ^ sign, gen=[any16]))
        cases.append(ir_case('%s_fabs_bits' % irty, ['u16'], 'u16',
                             '%%b = bitcast i16 %%a0 to %s\n%%n = call %s @llvm.fabs.%s(%s %%b)\n'
                             '%%r = bitcast %s %%n to i16' % (ty, ty, irty, ty, ty),
                             lambda i, v: v[0][i] & 0x7fff, {'declare %s @llvm.fabs.%s(%s)' % (ty, irty, ty)},
                             gen=[any16]))
    cases.append(ir_case('half_copysign_bits', ['u16', 'u16'], 'u16',
                         '%b = bitcast i16 %a0 to half\n%s = bitcast i16 %a1 to half\n'
                         '%n = call half @llvm.copysign.f16(half %b, half %s)\n%r = bitcast half %n to i16',
                         lambda i, v: v[0][i] & 0x7fff | v[1][i] & 0x8000,
                         {'declare half @llvm.copysign.f16(half, half)'}, [any16, any16]))
    # bfloat arithmetic.
    bfv = bf16_bits_to_f32
    for op in ('fadd', 'fsub', 'fmul', 'fdiv'):
        cases.append(ir_case('bf16_' + op, ['bf16', 'bf16'], 'bf16', '%%r = %s bfloat %%a0, %%a1' % op,
                             lambda i, v, op=op: bf16_bits(fp_op('bf16', op, bfv(v[0][i]), bfv(v[1][i]))),
                             gen=[bf16_gen(), bf16_gen(-4, 4)]))
    cases.append(ir_case('bf16_fma', ['bf16'] * 3, 'bf16',
                         '%r = call bfloat @llvm.fma.bf16(bfloat %a0, bfloat %a1, bfloat %a2)',
                         lambda i, v: bf16_bits(fma_ref('bf16', *(bfv(v[k][i]) for k in range(3)))),
                         {'declare bfloat @llvm.fma.bf16(bfloat, bfloat, bfloat)'},
                         [bf16_gen(-6, 6)] * 3))
    cases.append(ir_case('bf16_fcmp_olt', ['bf16', 'bf16'], 'u32',
                         '%c = fcmp olt bfloat %a0, %a1\n%r = zext i1 %c to i32',
                         lambda i, v: int(bfv(v[0][i]) < bfv(v[1][i])),
                         gen=[bf16_gen(-2, 2), lambda i, r: r.choice([r.randrange(1 << 16), 0x3f80, 0xbf80])]))
    cases.append(ir_case('sitofp_i32_bf16', ['i32'], 'bf16', '%r = sitofp i32 %a0 to bfloat',
                         lambda i, v: bf16_bits(round_to('bf16', v[0][i])), gen=[mag]))
    cases.append(ir_case('fptosi_bf16_i32', ['bf16'], 'i32', '%r = fptosi bfloat %a0 to i32',
                         lambda i, v: int(bfv(v[0][i])), gen=[bf16_gen(-3, 30)]))
    cases.append(ir_case('bf16_to_half', ['bf16'], 'f16',
                         '%f = fpext bfloat %a0 to float\n%r = fptrunc float %f to half',
                         lambda i, v: f16(bfv(v[0][i])), gen=[bf16_gen(-30, 20)], ulps=0, bitexact=True))
    return cases


I8_EDGE = [0, 1, -1, -128, 127, 2, -2, 64]
I16_EDGE = [0, 1, -1, -32768, 32767, 2, -2, 256]


def wrap(bits, signed):
    m = 1 << bits
    return (lambda x: (x + m // 2) % m - m // 2) if signed else (lambda x: x % m)


def group_llvm_int_narrow():
    """i8 and i16 arithmetic, division and compares; 64-bit division and
    overflow; the multiply-high idioms; i128; zero-poison counts; scmp/ucmp."""
    cases = []
    for bits, edge in ((8, I8_EDGE), (16, I16_EDGE)):
        it, sty, uty = 'i%d' % bits, 'i%d' % bits, 'u%d' % bits
        ws, wu = wrap(bits, True), wrap(bits, False)
        half = 1 << (bits - 1)
        amt = lambda i, r, b=bits: r.randrange(0, b)
        nz = lambda i, r, b=bits: r.randrange(1, 1 << b)
        snz = lambda i, r, h=half: r.choice([r.randrange(1, h), -r.randrange(1, h + 1),
                                             r.randrange(1, 10), -r.randrange(1, 10)])
        no_min = lambda i, r, h=half: r.randrange(-h + 1, h)
        for name, ty, inst, gens, ref in (
                ('add', sty, 'add', None, lambda a, b, w=ws: w(a + b)),
                ('sub', sty, 'sub', None, lambda a, b, w=ws: w(a - b)),
                ('mul', sty, 'mul', None, lambda a, b, w=ws: w(a * b)),
                ('shl', uty, 'shl', [None, amt], lambda a, b, w=wu: w(a << b)),
                ('lshr', uty, 'lshr', [None, amt], lambda a, b: a >> b),
                ('ashr', sty, 'ashr', [None, amt], lambda a, b: a >> b),
                ('udiv', uty, 'udiv', [None, nz], lambda a, b: a // b),
                ('urem', uty, 'urem', [None, nz], lambda a, b: a % b),
                ('sdiv', sty, 'sdiv', [no_min, snz], lambda a, b, w=ws: w(tdiv(a, b)[0])),
                ('srem', sty, 'srem', [no_min, snz], lambda a, b: tdiv(a, b)[1])):
            cases.append(ir_case('%s_%s' % (name, it), [ty, ty], ty, '%%r = %s %s %%a0, %%a1' % (inst, it),
                                 lambda i, v, ref=ref: ref(v[0][i], v[1][i]), gen=gens))
        m = (1 << bits) - 1
        for p, ref in (('slt', lambda a, b: a < b), ('sge', lambda a, b: a >= b),
                       ('ult', lambda a, b, m=m: a & m < b & m), ('ugt', lambda a, b, m=m: a & m > b & m),
                       ('eq', lambda a, b: a == b)):
            cases.append(ir_case('icmp_%s_%s' % (it, p), [sty, sty], 'u32',
                                 '%%c = icmp %s %s %%a0, %%a1\n%%r = zext i1 %%c to i32' % (p, it),
                                 lambda i, v, ref=ref: int(ref(v[0][i], v[1][i])),
                                 gen=[cyc(edge), cyc(edge, 8)]))
        for intr, ty, ref in (('smin', sty, min), ('smax', sty, max), ('umin', uty, min), ('umax', uty, max)):
            cases.append(int_case('%s_%s' % (intr, it), 'llvm.%s.%s' % (intr, it), ty, 2,
                                  lambda i, v, ref=ref: ref(v[0][i], v[1][i])))
        cases.append(int_case('abs_%s_min' % it, 'llvm.abs.' + it, sty, 1,
                              lambda i, v, w=ws: w(abs(v[0][i])), ', i1 false', cyc(edge)))
        cases.append(int_case('ctpop_' + it, 'llvm.ctpop.' + it, uty, 1, lambda i, v: bin(v[0][i]).count('1')))
        cases.append(int_case('ctlz_' + it, 'llvm.ctlz.' + it, uty, 1,
                              lambda i, v, b=bits: b - v[0][i].bit_length(), ', i1 false',
                              lambda i, r, b=bits: r.randrange(0, 1 << r.randrange(0, b + 1))))
        cases.append(int_case('cttz_' + it, 'llvm.cttz.' + it, uty, 1,
                              lambda i, v, b=bits: b if v[0][i] == 0 else (v[0][i] & -v[0][i]).bit_length() - 1,
                              ', i1 false', lambda i, r, b=bits: r.randrange(0, 1 << b) << r.randrange(0, b) & ((1 << b) - 1)))
        cases.append(int_case('bitreverse_' + it, 'llvm.bitreverse.' + it, uty, 1,
                              lambda i, v, b=bits: int(format(v[0][i], '0%db' % b)[::-1], 2)))
    # Zero-poison counts on nonzero inputs.
    for bits, ty in ((32, 'u32'), (64, 'u64')):
        nzg = lambda i, r, b=bits: r.randrange(1, 1 << r.randrange(1, b + 1)) << r.randrange(0, 4) & ((1 << b) - 1) or 1
        cases.append(int_case('ctlz_i%d_zero_poison' % bits, 'llvm.ctlz.i%d' % bits, ty, 1,
                              lambda i, v, b=bits: b - v[0][i].bit_length(), ', i1 true', nzg))
        cases.append(int_case('cttz_i%d_zero_poison' % bits, 'llvm.cttz.i%d' % bits, ty, 1,
                              lambda i, v: (v[0][i] & -v[0][i]).bit_length() - 1, ', i1 true', nzg))
    cases.append(int_case('abs_i32_min', 'llvm.abs.i32', 'i32', 1, lambda i, v: s32(abs(v[0][i])),
                          ', i1 false', cyc(I32_EDGE)))
    # Compares of extended narrow values.
    cases.append(ir_case('icmp_sext_i8_i32', ['i8', 'i32'], 'u32',
                         '%x = sext i8 %a0 to i32\n%c = icmp slt i32 %x, %a1\n%r = zext i1 %c to i32',
                         lambda i, v: int(v[0][i] < v[1][i]), gen=[None, lambda i, r: r.randrange(-200, 200)]))
    cases.append(ir_case('icmp_zext_i16_i32', ['u16', 'i32'], 'u32',
                         '%x = zext i16 %a0 to i32\n%c = icmp sgt i32 %x, %a1\n%r = zext i1 %c to i32',
                         lambda i, v: int(v[0][i] > v[1][i]), gen=[None, lambda i, r: r.randrange(-1000, 70000)]))
    # 64-bit division with full-width operands.
    s64g = lambda i, r: r.choice([-1, 1]) * r.randrange(1, 1 << r.randrange(1, 64))
    no_min64 = lambda i, r: r.randrange(-(1 << 63) + 1, 1 << 63)
    nz64 = lambda i, r: r.randrange(1, 1 << r.randrange(1, 65))
    for name, ty, inst, gens, ref in (
            ('sdiv_i64_full', 'i64', 'sdiv', [no_min64, s64g], lambda a, b: tdiv(a, b)[0]),
            ('srem_i64_full', 'i64', 'srem', [no_min64, s64g], lambda a, b: tdiv(a, b)[1]),
            ('udiv_i64_full', 'u64', 'udiv', [None, nz64], lambda a, b: a // b),
            ('urem_i64_full', 'u64', 'urem', [None, nz64], lambda a, b: a % b)):
        cases.append(ir_case(name, [ty, ty], ty, '%%r = %s i64 %%a0, %%a1' % inst,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]), gen=gens))
    cases.append(ir_case('srem_i32_pow2', ['i32'], 'i32', '%r = srem i32 %a0, 8',
                         lambda i, v: tdiv(v[0][i], 8)[1], gen=[mag]))
    cases.append(ir_case('sdiv_exact_i32', ['i32'], 'i32', '%m = mul i32 %a0, 12\n%r = sdiv exact i32 %m, 12',
                         lambda i, v: v[0][i], gen=[lambda i, r: r.randrange(-(1 << 27), 1 << 27)]))
    cases.append(ir_case('udiv_i64_c10', ['u64'], 'u64', '%r = udiv i64 %a0, 10', lambda i, v: v[0][i] // 10))
    # Multiply-high idioms.
    cases.append(ir_case('umulhi32_idiom', ['u32', 'u32'], 'u32',
                         '%x = zext i32 %a0 to i64\n%y = zext i32 %a1 to i64\n%m = mul i64 %x, %y\n'
                         '%h = lshr i64 %m, 32\n%r = trunc i64 %h to i32',
                         lambda i, v: v[0][i] * v[1][i] >> 32))
    cases.append(ir_case('smulhi32_idiom', ['i32', 'i32'], 'i32',
                         '%x = sext i32 %a0 to i64\n%y = sext i32 %a1 to i64\n%m = mul i64 %x, %y\n'
                         '%h = ashr i64 %m, 32\n%r = trunc i64 %h to i32',
                         lambda i, v: v[0][i] * v[1][i] >> 32))
    cases.append(ir_case('umulhi64_i128', ['u64', 'u64'], 'u64',
                         '%x = zext i64 %a0 to i128\n%y = zext i64 %a1 to i128\n%m = mul i128 %x, %y\n'
                         '%h = lshr i128 %m, 64\n%r = trunc i128 %h to i64',
                         lambda i, v: v[0][i] * v[1][i] >> 64))
    cases.append(ir_case('smulhi64_i128', ['i64', 'i64'], 'i64',
                         '%x = sext i64 %a0 to i128\n%y = sext i64 %a1 to i128\n%m = mul i128 %x, %y\n'
                         '%h = ashr i128 %m, 64\n%r = trunc i128 %h to i64',
                         lambda i, v: v[0][i] * v[1][i] >> 64))
    # i128: A = a1:a0 and B = a3:a2; even threads store the low half, odd the high.
    build = ('%xl = zext i64 %a0 to i128\n%xh0 = zext i64 %a1 to i128\n%xh = shl i128 %xh0, 64\n'
             '%A = or i128 %xh, %xl\n'
             '%yl = zext i64 %a2 to i128\n%yh0 = zext i64 %a3 to i128\n%yh = shl i128 %yh0, 64\n'
             '%B = or i128 %yh, %yl\n%sh = and i128 %B, 127\n')
    split = ('%lo = trunc i128 %R to i64\n%hs = lshr i128 %R, 64\n%hi = trunc i128 %hs to i64\n'
             '%odd = trunc i32 %tid to i1\n%r = select i1 %odd, i64 %hi, i64 %lo')
    M128 = (1 << 128) - 1

    def s128(x):
        x &= M128
        return x - (1 << 128) if x >> 127 else x
    for name, inst, ref, decl in (
            ('add', 'add i128 %A, %B', lambda a, b: a + b, None),
            ('sub', 'sub i128 %A, %B', lambda a, b: a - b, None),
            ('mul', 'mul i128 %A, %B', lambda a, b: a * b, None),
            ('xor', 'xor i128 %A, %B', lambda a, b: a ^ b, None),
            ('shl', 'shl i128 %A, %sh', lambda a, b: a << (b & 127), None),
            ('lshr', 'lshr i128 %A, %sh', lambda a, b: a >> (b & 127), None),
            ('ashr', 'ashr i128 %A, %sh', lambda a, b: s128(a) >> (b & 127), None),
            ('umin', 'call i128 @llvm.umin.i128(i128 %A, i128 %B)', min,
             'declare i128 @llvm.umin.i128(i128, i128)'),
            ('smax', 'call i128 @llvm.smax.i128(i128 %A, i128 %B)', lambda a, b: max(s128(a), s128(b)),
             'declare i128 @llvm.smax.i128(i128, i128)'),
            ('ctlz', 'call i128 @llvm.ctlz.i128(i128 %A, i1 false)', lambda a, b: 128 - a.bit_length(),
             'declare i128 @llvm.ctlz.i128(i128, i1)'),
            ('ult', 'icmp ult i128 %A, %B\n%R = zext i1 %Rc to i128', lambda a, b: int(a < b), None),
            ('slt', 'icmp slt i128 %A, %B\n%R = zext i1 %Rc to i128', lambda a, b: int(s128(a) < s128(b)),
             None)):
        body = build + ('%Rc = ' + inst if inst.startswith('icmp') else '%R = ' + inst) + '\n' + split

        def ref128(i, v, ref=ref):
            a = v[1][i] << 64 | v[0][i]
            b = v[3][i] << 64 | v[2][i]
            res = ref(a, b) & M128
            return res >> 64 if i & 1 else res & ((1 << 64) - 1)
        g = umag64 if name in ('mul', 'ctlz') else None
        cases.append(ir_case('i128_' + name, ['u64'] * 4, 'u64', body, ref128,
                             [decl] if decl else (), [g, g, g, g] if g else None))
    # Overflow intrinsics on i64: even threads store the value, odd the flag.
    for op, ty, ref in (('uadd', 'u64', lambda a, b: (a + b, a + b >= 1 << 64)),
                        ('usub', 'u64', lambda a, b: (a - b, a < b)),
                        ('umul', 'u64', lambda a, b: (a * b, a * b >= 1 << 64)),
                        ('smul', 'i64', lambda a, b: (a * b, not -(1 << 63) <= a * b < 1 << 63))):
        def ref_ovf(i, v, ref=ref):
            val, ovf = ref(v[0][i], v[1][i])
            return int(ovf) if i & 1 else u64(val)
        g = umag64 if ty == 'u64' else s64g
        cases.append(ir_case(op + '_with_overflow_i64', [ty, ty], 'u64',
                             '%%s = call { i64, i1 } @llvm.%s.with.overflow.i64(i64 %%a0, i64 %%a1)\n'
                             '%%v = extractvalue { i64, i1 } %%s, 0\n%%o = extractvalue { i64, i1 } %%s, 1\n'
                             '%%o64 = zext i1 %%o to i64\n%%odd = trunc i32 %%tid to i1\n'
                             '%%r = select i1 %%odd, i64 %%o64, i64 %%v' % op, ref_ovf,
                             {'declare { i64, i1 } @llvm.%s.with.overflow.i64(i64, i64)' % op}, [g, g]))
    for name, intr, ty, ref in (
            ('usub_sat_i64', 'llvm.usub.sat.i64', 'u64', lambda a, b: max(a - b, 0)),
            ('uadd_sat_i64', 'llvm.uadd.sat.i64', 'u64', lambda a, b: min(a + b, (1 << 64) - 1)),
            ('ssub_sat_i8', 'llvm.ssub.sat.i8', 'i8', lambda a, b: max(-128, min(127, a - b)))):
        cases.append(int_case(name, intr, ty, 2, lambda i, v, ref=ref: ref(v[0][i], v[1][i]),
                              gen=umag64 if ty == 'u64' else None))
    cases.append(int_case('rotl_i64_fshl', 'llvm.fshl.i64', 'u64', 3,
                          lambda i, v: u64(v[0][i] << v[2][i] % 64 | v[0][i] >> (64 - v[2][i] % 64)),
                          gen=None))
    cases[-1].body = '%r = call i64 @llvm.fshl.i64(i64 %a0, i64 %a0, i64 %a2)'
    # Three-way compares (Rust's Ord::cmp) and compare/select chains.
    for name, ty in (('scmp', 'i32'), ('ucmp', 'u32')):
        cases.append(ir_case(name + '_i32', [ty, ty], 'i32',
                             '%%c = call i8 @llvm.%s.i8.i32(i32 %%a0, i32 %%a1)\n%%r = sext i8 %%c to i32' % name,
                             lambda i, v: (v[0][i] > v[1][i]) - (v[0][i] < v[1][i]),
                             {'declare i8 @llvm.%s.i8.i32(i32, i32)' % name},
                             gen=[cyc([u32(x) if ty == 'u32' else x for x in I32_EDGE]),
                                  cyc([u32(x) if ty == 'u32' else x for x in I32_EDGE], 8)]))
    cases.append(ir_case('clamp_select_chain', ['i32'], 'i32',
                         '%lt = icmp slt i32 %a0, -1000\n%x = select i1 %lt, i32 -1000, i32 %a0\n'
                         '%gt = icmp sgt i32 %x, 1000\n%r = select i1 %gt, i32 1000, i32 %x',
                         lambda i, v: max(-1000, min(1000, v[0][i])), gen=[mag]))
    cases.append(ir_case('sign_idiom', ['i32'], 'i32',
                         '%p = icmp sgt i32 %a0, 0\n%n = icmp slt i32 %a0, 0\n%pz = zext i1 %p to i32\n'
                         '%nz = zext i1 %n to i32\n%r = sub i32 %pz, %nz',
                         lambda i, v: (v[0][i] > 0) - (v[0][i] < 0), gen=[cyc(I32_EDGE)]))
    cases.append(ir_case('logical_and_or_i1', ['u32', 'u32'], 'u32',
                         '%x = icmp ugt i32 %a0, 2147483648\n%y = icmp ult i32 %a1, 1073741824\n'
                         '%and = select i1 %x, i1 %y, i1 false\n%or = select i1 %x, i1 true, i1 %y\n'
                         '%az = zext i1 %and to i32\n%oz = zext i1 %or to i32\n%o2 = shl i32 %oz, 1\n'
                         '%r = or i32 %az, %o2',
                         lambda i, v: int(v[0][i] > 1 << 31 and v[1][i] < 1 << 30) |
                         int(v[0][i] > 1 << 31 or v[1][i] < 1 << 30) << 1))
    return cases


def lanes_u8(x, n=4):
    return [x >> 8 * k & 0xff for k in range(n)]


def from_u8(bs):
    return sum((b & 0xff) << 8 * k for k, b in enumerate(bs))


def vec_case(name, n, ty, body, ref, decls=(), gen=None, ulps=None, bitexact=False, out_init=None,
             in_ty=None, out_ty=None, align=None):
    """Thread i < N/n loads a <n x ty> from each input at i, the body maps
    the vectors %x0, %x1, ... to %y, a <n x ty>, and the thread stores %y at
    out[i]; ref(i, v) is the value of element i of out."""
    in_ty = in_ty or ty
    out_ty = out_ty or ty
    nin = len(gen) if gen else 1
    ir_in, ir_out = TYPES[in_ty][0], TYPES[out_ty][0]
    a = ', align %d' % align if align else ''
    loads = ''.join('%%px%d = getelementptr <%d x %s>, ptr addrspace(1) %%in%d, i32 %%tid\n'
                    '%%x%d = load <%d x %s>, ptr addrspace(1) %%px%d%s\n' % (k, n, ir_in, k, k, n, ir_in, k, a)
                    for k in range(nin))
    c = full_case(name, [in_ty] * nin, out_ty, ref,
                  '%%go = icmp ult i32 %%tid, %d\nbr i1 %%go, label %%body, label %%done\nbody:\n' % (N // n) +
                  loads + body.strip() + '\n'
                  '%%po = getelementptr <%d x %s>, ptr addrspace(1) %%out, i32 %%tid\n'
                  'store <%d x %s> %%y, ptr addrspace(1) %%po%s\nbr label %%done\ndone:\nret void'
                  % (n, ir_out, n, ir_out, a), decls, gen or [None], ulps)
    c.bitexact = bitexact
    c.out_init = out_init
    return c


def group_llvm_vector_ext():
    """Odd and small vector types in memory, per-lane shifts, vector compares
    and conversions, splats, masks from memory, more reductions, phis of
    vectors, narrow-element vectors."""
    cases = []
    spread = lambda i, r: r.uniform(-1, 1) * 10.0 ** r.randrange(-3, 6)
    # <2 x float> (no contraction of the fmul and fadd).
    cases.append(vec_case('v2f32_mul_add', 2, 'f32',
                          '%m = fmul <2 x float> %x0, %x1\n%y = fadd <2 x float> %m, <float 1.0, float -2.0>',
                          lambda i, v: f32(f32(v[0][i] * v[1][i]) + (1.0 if i % 2 == 0 else -2.0)),
                          gen=[spread, spread], ulps=0))
    # <3 x float> at its allocation stride (16 bytes): the fourth float of each
    # slot is padding, which the store must not write.
    def v3_ref(i, v):
        return 7.0 if i % 4 == 3 else f32(v[0][i] * (2.0, 3.0, 0.5)[i % 4])
    c = vec_case('v3f32_stride16', 4, 'f32', '%y = fmul <4 x float> %x0, <float 2.0, float 3.0, float 0.5>',
                 v3_ref, gen=[spread], ulps=0, out_init=[7.0] * N)
    c.full = c.full.replace('<4 x float>', '<3 x float>')
    cases.append(c)
    # <3 x i32> packed (12-byte stride, align 4), as a packed_int3 array.
    c = full_case('v3i32_packed', ['u32'], 'u32',
                  lambda i, v: 0xdeadbeef if i >= 63 else u32(v[0][i] + i % 3 + 1),
                  '%go = icmp ult i32 %tid, 21\nbr i1 %go, label %body, label %done\nbody:\n'
                  '%off = mul i32 %tid, 12\n'
                  '%pa = getelementptr i8, ptr addrspace(1) %in0, i32 %off\n'
                  '%x = load <3 x i32>, ptr addrspace(1) %pa, align 4\n'
                  '%y = add <3 x i32> %x, <i32 1, i32 2, i32 3>\n'
                  '%po = getelementptr i8, ptr addrspace(1) %out, i32 %off\n'
                  'store <3 x i32> %y, ptr addrspace(1) %po, align 4\nbr label %done\ndone:\nret void')
    c.out_init = [0xdeadbeef] * N
    cases.append(c)
    c = full_case('v3f32_extract_dynamic', ['f32'], 'f32',
                  lambda i, v: v[0][i // 3 * 3 + i % 3] if i < 63 else 0.0,
                  '%go = icmp ult i32 %tid, 63\nbr i1 %go, label %body, label %done\nbody:\n'
                  '%q = udiv i32 %tid, 3\n%k = urem i32 %tid, 3\n%off = mul i32 %q, 12\n'
                  '%pa = getelementptr i8, ptr addrspace(1) %in0, i32 %off\n'
                  '%x = load <3 x float>, ptr addrspace(1) %pa, align 4\n'
                  '%r = extractelement <3 x float> %x, i32 %k\n'
                  '%po = getelementptr float, ptr addrspace(1) %out, i32 %tid\n'
                  'store float %r, ptr addrspace(1) %po\nbr label %done\ndone:\nret void', ulps=0)
    c.out_init = [0.0] * N
    cases.append(c)
    # <8 x i8> lanes of two words.
    cases.append(vec_case('v8i8_mul_add', 8, 'u8',
                          '%m = mul <8 x i8> %x0, %x1\n%y = add <8 x i8> %m, splat (i8 7)',
                          lambda i, v: (v[0][i] * v[1][i] + 7) & 0xff, gen=[None, None]))
    cases.append(vec_case('v4i8_shl_per_lane', 4, 'u8',
                          '%y = shl <4 x i8> %x0, <i8 0, i8 3, i8 5, i8 7>',
                          lambda i, v: v[0][i] << (0, 3, 5, 7)[i % 4] & 0xff))
    cases.append(vec_case('v4i8_lshr_var', 4, 'u8',
                          '%s = and <4 x i8> %x1, splat (i8 7)\n%y = lshr <4 x i8> %x0, %s',
                          lambda i, v: v[0][i] >> (v[1][i] & 7), gen=[None, None]))
    cases.append(vec_case('v4i8_ashr_var', 4, 'i8',
                          '%s = and <4 x i8> %x1, splat (i8 7)\n%y = ashr <4 x i8> %x0, %s',
                          lambda i, v: v[0][i] >> (v[1][i] & 7), gen=[None, None]))
    cases.append(vec_case('v4i32_shifts_per_lane', 4, 'u32',
                          '%s = lshr <4 x i32> %x1, splat (i32 27)\n%a = shl <4 x i32> %x0, %s\n'
                          '%b = lshr <4 x i32> %x0, %s\n%c = ashr <4 x i32> %x0, %s\n'
                          '%ab = xor <4 x i32> %a, %b\n%y = add <4 x i32> %ab, %c',
                          lambda i, v: u32((v[0][i] << (v[1][i] >> 27)) ^ (v[0][i] >> (v[1][i] >> 27)))
                          + u32(s32(v[0][i]) >> (v[1][i] >> 27)) & 0xffffffff, gen=[None, None]))
    cases.append(vec_case('v2i16_uadd_sat', 2, 'u16',
                          '%y = call <2 x i16> @llvm.uadd.sat.v2i16(<2 x i16> %x0, <2 x i16> %x1)',
                          lambda i, v: min(v[0][i] + v[1][i], 0xffff),
                          {'declare <2 x i16> @llvm.uadd.sat.v2i16(<2 x i16>, <2 x i16>)'}, [None, None]))
    cases.append(vec_case('v4i8_usub_sat', 4, 'u8',
                          '%y = call <4 x i8> @llvm.usub.sat.v4i8(<4 x i8> %x0, <4 x i8> %x1)',
                          lambda i, v: max(v[0][i] - v[1][i], 0),
                          {'declare <4 x i8> @llvm.usub.sat.v4i8(<4 x i8>, <4 x i8>)'}, [None, None]))
    cases.append(vec_case('v4i8_smin', 4, 'i8',
                          '%y = call <4 x i8> @llvm.smin.v4i8(<4 x i8> %x0, <4 x i8> %x1)',
                          lambda i, v: min(v[0][i], v[1][i]),
                          {'declare <4 x i8> @llvm.smin.v4i8(<4 x i8>, <4 x i8>)'}, [None, None]))
    cases.append(vec_case('v4i32_ctpop', 4, 'u32',
                          '%y = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %x0)',
                          lambda i, v: bin(v[0][i]).count('1'), {'declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)'}))
    cases.append(vec_case('v2i32_bswap', 2, 'u32',
                          '%y = call <2 x i32> @llvm.bswap.v2i32(<2 x i32> %x0)',
                          lambda i, v: int.from_bytes(v[0][i].to_bytes(4, 'little'), 'big'),
                          {'declare <2 x i32> @llvm.bswap.v2i32(<2 x i32>)'}))
    cases.append(vec_case('v2i64_mul_lshr', 2, 'u64',
                          '%m = mul <2 x i64> %x0, %x1\n%s = and <2 x i64> %x1, splat (i64 63)\n'
                          '%y = lshr <2 x i64> %m, %s',
                          lambda i, v: u64(v[0][i] * v[1][i]) >> (v[1][i] & 63), gen=[None, None]))
    cases.append(vec_case('v2i64_icmp_slt', 2, 'i64',
                          '%c = icmp slt <2 x i64> %x0, %x1\n%y = sext <2 x i1> %c to <2 x i64>',
                          lambda i, v: -int(v[0][i] < v[1][i]), gen=[None, None]))
    # Vector float compares, selects and conversions.
    cases.append(vec_case('v4f32_fcmp_select', 4, 'f32',
                          '%c = fcmp olt <4 x float> %x0, %x1\n%n = fneg <4 x float> %x0\n'
                          '%y = select <4 x i1> %c, <4 x float> %x0, <4 x float> %n',
                          lambda i, v: v[0][i] if v[0][i] < v[1][i] else -v[0][i],
                          gen=[edge_a, edge_b], ulps=0, bitexact=True))
    cases.append(vec_case('v4f32_fcmp_uno', 4, 'f32', '%c = fcmp uno <4 x float> %x0, %x1\n'
                          '%y = uitofp <4 x i1> %c to <4 x float>',
                          lambda i, v: float(v[0][i] != v[0][i] or v[1][i] != v[1][i]),
                          gen=[edge_a, edge_b], ulps=0))
    cases.append(vec_case('v4f32_fptosi', 4, 'f32', '%y = fptosi <4 x float> %x0 to <4 x i32>',
                          lambda i, v: int(v[0][i]), gen=[lambda i, r: r.uniform(-1e6, 1e6)], out_ty='i32'))
    cases.append(vec_case('v4i32_sitofp', 4, 'i32', '%y = sitofp <4 x i32> %x0 to <4 x float>',
                          lambda i, v: int_to_f32(v[0][i]), gen=[mag], out_ty='f32', ulps=0))
    cases.append(vec_case('v4i32_uitofp', 4, 'u32', '%y = uitofp <4 x i32> %x0 to <4 x float>',
                          lambda i, v: int_to_f32(v[0][i]), out_ty='f32', ulps=0))
    cases.append(vec_case('v4f16_fpext', 4, 'f16', '%y = fpext <4 x half> %x0 to <4 x float>',
                          lambda i, v: v[0][i], gen=[cyc(F16_EDGE1)], out_ty='f32', ulps=0, bitexact=True))
    cases.append(vec_case('v2f32_fptrunc', 2, 'f32', '%y = fptrunc <2 x float> %x0 to <2 x half>',
                          lambda i, v: f16(v[0][i]), gen=[wide16], out_ty='f16', ulps=0, bitexact=True))
    for name, ins, ref in (('fabs', 'call <4 x float> @llvm.fabs.v4f32(<4 x float> %x0)',
                            lambda x: x if x != x else abs(x)),
                           ('fneg', 'fneg <4 x float> %x0', lambda x: -x)):
        cases.append(vec_case('v4f32_' + name, 4, 'f32', '%y = ' + ins, lambda i, v, ref=ref: ref(v[0][i]),
                              {'declare <4 x float> @llvm.fabs.v4f32(<4 x float>)'} if name == 'fabs' else (),
                              [edge1], 0, True))
    cases.append(vec_case('v4f32_copysign', 4, 'f32',
                          '%y = call <4 x float> @llvm.copysign.v4f32(<4 x float> %x0, <4 x float> %x1)',
                          lambda i, v: math.copysign(v[0][i], v[1][i]),
                          {'declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>)'},
                          [edge_a, edge_b], 0, True))
    cases.append(vec_case('v4f32_minnum', 4, 'f32',
                          '%y = call <4 x float> @llvm.minnum.v4f32(<4 x float> %x0, <4 x float> %x1)',
                          lambda i, v: minnum(v[0][i], v[1][i]),
                          {'declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)'},
                          [edge_a, edge_b], 0))
    cases.append(vec_case('v2f16_maxnum', 2, 'f16',
                          '%y = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %x0, <2 x half> %x1)',
                          lambda i, v: maxnum(v[0][i], v[1][i]),
                          {'declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>)'},
                          [cyc(F16_EDGE), cyc(F16_EDGE, 8)], 0))
    # Splat through shufflevector zeroinitializer.
    cases.append(ir_case('vec_splat_shuffle', ['u32'], 'u32',
                         '%s0 = insertelement <4 x i32> poison, i32 %a0, i64 0\n'
                         '%s = shufflevector <4 x i32> %s0, <4 x i32> poison, <4 x i32> zeroinitializer\n'
                         '%m = mul <4 x i32> %s, <i32 1, i32 2, i32 3, i32 4>\n%k = and i32 %tid, 3\n'
                         '%r = extractelement <4 x i32> %m, i32 %k', lambda i, v: u32(v[0][i] * (i % 4 + 1))))
    # A vector loaded from device memory, element chosen per thread.
    cases.append(full_case('vec_load_extract_dynamic', ['u32'], 'u32', lambda i, v: u32(v[0][i] + 1),
                           '%q = lshr i32 %tid, 2\n%k = and i32 %tid, 3\n'
                           '%pv = getelementptr <4 x i32>, ptr addrspace(1) %in0, i32 %q\n'
                           '%x = load <4 x i32>, ptr addrspace(1) %pv\n'
                           '%e = extractelement <4 x i32> %x, i32 %k\n%r = add i32 %e, 1\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store i32 %r, ptr addrspace(1) %po\nret void'))
    # Byte reversal of a <16 x i8>.
    cases.append(ir_case('v16i8_shuffle_reverse', ['u32'] * 4, 'u32', VEC4_I32 +
                         '%b = bitcast <4 x i32> %v to <16 x i8>\n'
                         '%s = shufflevector <16 x i8> %b, <16 x i8> poison, <16 x i32> <' +
                         ', '.join('i32 %d' % (15 - k) for k in range(16)) + '>\n' +
                         '%w = bitcast <16 x i8> %s to <4 x i32>\n%r = extractelement <4 x i32> %w, i32 %k',
                         lambda i, v: int.from_bytes(v[3 - i % 4][i].to_bytes(4, 'little'), 'big')))
    # Masks: an i8 as <8 x i1>, a <8 x i1> compare as an i8 (movemask).
    cases.append(ir_case('vec_mask_from_i8', ['u8', 'u32', 'u32'], 'u32',
                         '%m = bitcast i8 %a0 to <8 x i1>\n%x0 = insertelement <2 x i32> poison, i32 %a1, i32 0\n'
                         '%x1 = insertelement <2 x i32> %x0, i32 %a2, i32 1\n%x = bitcast <2 x i32> %x1 to <8 x i8>\n'
                         '%s = select <8 x i1> %m, <8 x i8> %x, <8 x i8> splat (i8 -1)\n'
                         '%w = bitcast <8 x i8> %s to <2 x i32>\n%k = and i32 %tid, 1\n'
                         '%r = extractelement <2 x i32> %w, i32 %k',
                         lambda i, v: from_u8([b if v[0][i] >> (4 * (i & 1) + j) & 1 else 0xff
                                               for j, b in enumerate(lanes_u8(v[1 + (i & 1)][i]))])))
    cases.append(ir_case('vec_movemask_i8', ['u32', 'u32'], 'u8',
                         '%x0 = insertelement <2 x i32> poison, i32 %a0, i32 0\n'
                         '%x = insertelement <2 x i32> %x0, i32 %a1, i32 1\n%b = bitcast <2 x i32> %x to <8 x i8>\n'
                         '%c = icmp slt <8 x i8> %b, zeroinitializer\n%r = bitcast <8 x i1> %c to i8',
                         lambda i, v: sum(1 << k for k, b in enumerate(lanes_u8(v[0][i]) + lanes_u8(v[1][i]))
                                          if b >= 128)))
    cases.append(ir_case('vec_bool_bytes_select', ['u32', 'u32'], 'u32',
                         '%b = bitcast i32 %a0 to <4 x i8>\n%m = icmp ne <4 x i8> %b, zeroinitializer\n'
                         '%x = bitcast i32 %a1 to <4 x i8>\n%n = sub <4 x i8> zeroinitializer, %x\n'
                         '%s = select <4 x i1> %m, <4 x i8> %x, <4 x i8> %n\n%r = bitcast <4 x i8> %s to i32',
                         lambda i, v: from_u8([x if m else -x for m, x in zip(lanes_u8(v[0][i]), lanes_u8(v[1][i]))]),
                         gen=[lambda i, r: r.choice([0, 1, 0x100, 0x10001, 0x01000100, r.randrange(1 << 32)]), None]))
    # Integer vector casts.
    cases.append(ir_case('vec_trunc_v4i32_v4i8', ['u32'] * 4, 'u32', VEC4_I32 +
                         '%tr = trunc <4 x i32> %v to <4 x i8>\n%r = bitcast <4 x i8> %tr to i32',
                         lambda i, v: from_u8([v[k][i] for k in range(4)])))
    cases.append(ir_case('vec_sext_v4i8_v4i32', ['u32'], 'i32',
                         '%b = bitcast i32 %a0 to <4 x i8>\n%x = sext <4 x i8> %b to <4 x i32>\n'
                         '%k = and i32 %tid, 3\n%r = extractelement <4 x i32> %x, i32 %k',
                         lambda i, v: s8(lanes_u8(v[0][i])[i % 4])))
    cases.append(ir_case('vec_zext_v2i16_v2i64', ['u32'], 'u64',
                         '%b = bitcast i32 %a0 to <2 x i16>\n%x = zext <2 x i16> %b to <2 x i64>\n'
                         '%k = and i32 %tid, 1\n%r = extractelement <2 x i64> %x, i32 %k',
                         lambda i, v: v[0][i] >> 16 * (i % 2) & 0xffff))
    # A loop-carried vector phi.
    def phi_ref(i, v):
        acc = [0] * 4
        for _ in range(v[0][i] & 7):
            acc = [u32(a * 3 + v[k][i]) for k, a in enumerate(acc)]
        return acc[i % 4]
    cases.append(ir_case('vec_phi_loop', ['u32'] * 4, 'u32', VEC4_I32 +
                         '%n = and i32 %a0, 7\nbr label %h\n'
                         'h:\n%j = phi i32 [ 0, %entry ], [ %j1, %b ]\n'
                         '%acc = phi <4 x i32> [ zeroinitializer, %entry ], [ %acc1, %b ]\n'
                         '%more = icmp ult i32 %j, %n\nbr i1 %more, label %b, label %x\n'
                         'b:\n%m3 = mul <4 x i32> %acc, splat (i32 3)\n%acc1 = add <4 x i32> %m3, %v\n'
                         '%j1 = add i32 %j, 1\nbr label %h\n'
                         'x:\n%r = extractelement <4 x i32> %acc, i32 %k', phi_ref))
    # More reductions.
    U4 = ['u32'] * 4
    for op, ty, ref in (('or', 'u32', lambda xs: xs[0] | xs[1] | xs[2] | xs[3]),
                        ('smin', 'i32', min), ('umin', 'u32', min), ('smax', 'i32', max)):
        cases.append(ir_case('vec_reduce_%s_v4' % op, [ty] * 4, ty,
                             VEC4_I32 + '%%r = call i32 @llvm.vector.reduce.%s.v4i32(<4 x i32> %%v)' % op,
                             lambda i, v, ref=ref: ref([v[k][i] for k in range(4)]),
                             {'declare i32 @llvm.vector.reduce.%s.v4i32(<4 x i32>)' % op}))
    for op, ref in (('or', any), ('and', all)):
        cases.append(ir_case('vec_reduce_%s_v4i1' % op, U4, 'u32',
                             VEC4_I32 + '%%c = icmp ugt <4 x i32> %%v, splat (i32 1073741824)\n'
                             '%%b = call i1 @llvm.vector.reduce.%s.v4i1(<4 x i1> %%c)\n%%r = zext i1 %%b to i32' % op,
                             lambda i, v, ref=ref: int(ref(v[k][i] > 1 << 30 for k in range(4))),
                             {'declare i1 @llvm.vector.reduce.%s.v4i1(<4 x i1>)' % op}))
    F4 = ['f32'] * 4
    nanny = lambda i, r: math.nan if r.randrange(6) == 0 else r.uniform(-100, 100)

    def ordered_mul(i, v):
        acc = 1.0
        for k in range(4):
            acc = f32(acc * v[k][i])
        return acc
    cases.append(ir_case('vec_reduce_fmul_ordered', F4, 'f32',
                         VEC4_F32 + '%r = call float @llvm.vector.reduce.fmul.v4f32(float 1.0, <4 x float> %v)',
                         ordered_mul, {'declare float @llvm.vector.reduce.fmul.v4f32(float, <4 x float>)'},
                         [spread] * 4, 0))
    cases.append(ir_case('vec_reduce_fmin_nan', F4, 'f32',
                         VEC4_F32 + '%r = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> %v)',
                         lambda i, v: functools.reduce(minnum, [v[k][i] for k in range(4)]),
                         {'declare float @llvm.vector.reduce.fmin.v4f32(<4 x float>)'}, [nanny] * 4, 0))
    cases.append(ir_case('vec_reduce_fmaximum_nan', F4, 'f32',
                         VEC4_F32 + '%r = call float @llvm.vector.reduce.fmaximum.v4f32(<4 x float> %v)',
                         lambda i, v: functools.reduce(maximum, [v[k][i] for k in range(4)]),
                         {'declare float @llvm.vector.reduce.fmaximum.v4f32(<4 x float>)'}, [nanny] * 4, 0))
    return cases


def c_pow(x, y):
    """C's pow (LLVM's llvm.pow)."""
    def odd_int(t):
        return math.isfinite(t) and t == int(t) and int(t) % 2 == 1
    try:
        return math.pow(x, y)
    except ValueError:
        if x == 0 and y < 0:
            return math.copysign(math.inf, x) if odd_int(y) else math.inf
        return math.nan
    except OverflowError:
        return -math.inf if x < 0 and odd_int(y) else math.inf


def c_log(fn):
    def r(x):
        if x != x or x == math.inf:
            return x
        if x == 0:
            return -math.inf
        if x < 0:
            return math.nan
        return fn(x)
    return r


def c_exp(fn, big):
    def r(x):
        if x != x:
            return x
        if x == math.inf or x > big:
            return math.inf
        if x == -math.inf:
            return 0.0
        return fn(x)
    return r


MATH_EDGE = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 0.5, 128.0, -150.0, -126.0, 2.0,
             1e-30, 100.0, -100.0, 89.0, -104.0, 3.0, 0.25, 10.0, -0.5, 1e30]


def group_llvm_math_ext():
    """fma on special values and its single rounding, the 1/sqrt idioms,
    powi, lround/lrint, canonicalize, minimumnum/maximumnum, sincos, the
    special values of pow, exp and log, half math."""
    cases = []
    fma_decl = {'declare float @llvm.fma.f32(float, float, float)'}
    fma_body = '%r = call float @llvm.fma.f32(float %a0, float %a1, float %a2)'
    cases.append(ir_case('fma_f32_edges', ['f32'] * 3, 'f32', fma_body,
                         lambda i, v: fma_ref('f32', v[0][i], v[1][i], v[2][i]), fma_decl,
                         [edge_a, edge_b, lambda i, r: F32_EDGE1[i * 7 % len(F32_EDGE1)]], 0, True))

    def fused(k):
        def g(i, r):
            rr = random.Random(i)
            a = f32(rr.uniform(1, 2) * 2.0 ** rr.randrange(-20, 20))
            b = f32(rr.uniform(-2, 2) * 2.0 ** rr.randrange(-20, 20))
            return (a, b, -f32(a * b))[k]
        return g
    cases.append(ir_case('fma_f32_single_rounding', ['f32'] * 3, 'f32', fma_body,
                         lambda i, v: fma_ref('f32', v[0][i], v[1][i], v[2][i]), fma_decl,
                         [fused(0), fused(1), fused(2)], 0, True, source='fma(a, b, -(a*b)) is the error of a*b'))
    wide = lambda i, r: r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randrange(-60, 60)
    cases.append(ir_case('fma_f32_rand_exact', ['f32'] * 3, 'f32', fma_body,
                         lambda i, v: fma_ref('f32', v[0][i], v[1][i], v[2][i]), fma_decl,
                         [wide] * 3, 0, True))
    pos_ = lambda i, r: r.uniform(1, 2) * 2.0 ** r.randrange(-30, 30)
    cases.append(ir_case('rsqrt_idiom', ['f32'], 'f32',
                         '%s = call float @llvm.sqrt.f32(float %a0)\n%r = fdiv float 1.0, %s',
                         lambda i, v: f32(1 / f32(math.sqrt(v[0][i]))), {'declare float @llvm.sqrt.f32(float)'},
                         [pos_], 0))
    c = ir_case('rsqrt_idiom_afn', ['f32'], 'f32',
                '%s = call afn float @llvm.sqrt.f32(float %a0)\n%r = fdiv afn float 1.0, %s',
                lambda i, v: 1 / math.sqrt(v[0][i]), {'declare float @llvm.sqrt.f32(float)'}, [pos_], 16)
    cases.append(c)
    cases.append(ir_case('fdiv_arcp', ['f32', 'f32'], 'f32', '%r = fdiv arcp float %a0, %a1',
                         lambda i, v: v[0][i] / v[1][i], gen=[pos_, pos_], ulps=2))
    cases.append(ir_case('fdiv_afn', ['f32', 'f32'], 'f32', '%r = fdiv afn float %a0, %a1',
                         lambda i, v: v[0][i] / v[1][i], gen=[pos_, pos_], ulps=4))
    cases.append(ir_case('powi_f32', ['f32', 'i32'], 'f32',
                         '%r = call float @llvm.powi.f32.i32(float %a0, i32 %a1)',
                         lambda i, v: v[0][i] ** v[1][i], {'declare float @llvm.powi.f32.i32(float, i32)'},
                         [lambda i, r: r.uniform(0.5, 2) * r.choice([-1, 1]), lambda i, r: r.randrange(-8, 9)],
                         16))
    ties = lambda i, r: r.choice([r.randrange(-1000, 1000) + 0.5, r.uniform(-1e6, 1e6), r.randrange(-50, 50) * 0.25])
    rne = lambda x: int(round(x))
    away = lambda x: int(math.copysign(math.floor(abs(x) + 0.5), x))
    for name, ty, ref in (('lround.i32', 'i32', away), ('llround.i64', 'i64', away),
                          ('lrint.i32', 'i32', rne), ('llrint.i64', 'i64', rne)):
        irty = TYPES[ty][0]
        cases.append(ir_case(name.replace('.', '_'), ['f32'], ty,
                             '%%r = call %s @llvm.%s.f32(float %%a0)' % (irty, name),
                             lambda i, v, ref=ref: ref(v[0][i]),
                             {'declare %s @llvm.%s.f32(float)' % (irty, name)}, [ties]))
    cases.append(ir_case('canonicalize_f32', ['f32'], 'f32',
                         '%r = call float @llvm.canonicalize.f32(float %a0)', lambda i, v: v[0][i],
                         {'declare float @llvm.canonicalize.f32(float)'}, [edge1], 0, True))
    for name, ref in (('minimumnum', lambda a, b: minimum(a, b) if a == a and b == b else minnum(a, b)),
                      ('maximumnum', lambda a, b: maximum(a, b) if a == a and b == b else maxnum(a, b))):
        cases.append(ir_case(name, ['f32', 'f32'], 'f32',
                             '%%r = call float @llvm.%s.f32(float %%a0, float %%a1)' % name,
                             lambda i, v, ref=ref: ref(v[0][i], v[1][i]),
                             {'declare float @llvm.%s.f32(float, float)' % name}, [edge_a, edge_b], 0, True))
    cases.append(ir_case('llvm_sincos', ['f32'], 'f32',
                         '%sc = call { float, float } @llvm.sincos.f32(float %a0)\n'
                         '%sn = extractvalue { float, float } %sc, 0\n%cs = extractvalue { float, float } %sc, 1\n'
                         '%c2 = fmul float %cs, 2.0\n%r = fadd float %sn, %c2',
                         lambda i, v: math.sin(v[0][i]) + 2 * math.cos(v[0][i]),
                         {'declare { float, float } @llvm.sincos.f32(float)'}, ulps=8))
    cases[-1].abstol = 1e-6
    # Special values of the precise functions (C99 Annex F).
    for fn, ref in (('exp', c_exp(math.exp, 89)), ('exp2', c_exp(lambda x: 2.0 ** x, 128)),
                    ('log', c_log(math.log)), ('log2', c_log(math.log2))):
        cases.append(ir_case('edge_' + fn, ['f32'], 'f32', '%%r = call float @llvm.%s.f32(float %%a0)' % fn,
                             lambda i, v, ref=ref: ref(v[0][i]), {'declare float @llvm.%s.f32(float)' % fn},
                             [cyc(MATH_EDGE)], 4))
    pow_x = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -2.0, 0.5]
    pow_y = [0.0, math.inf, -math.inf, math.nan, 3.0, -1.0, 0.5, -2.0]
    cases.append(ir_case('edge_pow', ['f32', 'f32'], 'f32', '%r = call float @llvm.pow.f32(float %a0, float %a1)',
                         lambda i, v: c_pow(v[0][i], v[1][i]), {'declare float @llvm.pow.f32(float, float)'},
                         [cyc(pow_x), cyc(pow_y, 8)], 16))
    cases.append(ir_case('pow_negative_base', ['f32', 'f32'], 'f32',
                         '%r = call float @llvm.pow.f32(float %a0, float %a1)',
                         lambda i, v: c_pow(v[0][i], v[1][i]), {'declare float @llvm.pow.f32(float, float)'},
                         [lambda i, r: -r.uniform(0.5, 4), lambda i, r: float(r.randrange(-6, 7))], 16))
    # half math (bounded results; Metal's half functions are accurate to a few half ULPs).
    for fn, gen, ref in (('exp', lambda i, r: r.uniform(-4, 1), math.exp),
                         ('log', lambda i, r: r.uniform(0.5, 4), math.log),
                         ('cos', lambda i, r: r.uniform(-4, 4), math.cos),
                         ('tanh', lambda i, r: r.uniform(-3, 3), math.tanh)):
        c = ir_case('f16_llvm_' + fn, ['f16'], 'f16', '%%r = call half @llvm.%s.f16(half %%a0)' % fn,
                    lambda i, v, ref=ref: f16(ref(v[0][i])), {'declare half @llvm.%s.f16(half)' % fn}, [gen])
        c.abstol = 2.0 ** -7
        cases.append(c)
    c = ir_case('f16_llvm_pow', ['f16', 'f16'], 'f16', '%r = call half @llvm.pow.f16(half %a0, half %a1)',
                lambda i, v: f16(v[0][i] ** v[1][i]), {'declare half @llvm.pow.f16(half, half)'},
                [lambda i, r: r.uniform(0.5, 2), lambda i, r: r.uniform(-2, 2)])
    c.abstol = 2.0 ** -6
    cases.append(c)
    return cases


CF2_HELPERS = '''define internal { i32, i32 } @cf2_divmod(i32 %a, i32 %b, ptr %rem) noinline {
entry:
  %q = udiv i32 %a, %b
  %m = urem i32 %a, %b
  store i32 %m, ptr %rem
  %p0 = insertvalue { i32, i32 } poison, i32 %q, 0
  %p = insertvalue { i32, i32 } %p0, i32 %m, 1
  ret { i32, i32 } %p
}'''

CF2_VEC = '''define internal <4 x float> @cf2_scale(<4 x float> %v, float %s) noinline {
entry:
  %i = insertelement <4 x float> poison, float %s, i64 0
  %sp = shufflevector <4 x float> %i, <4 x float> poison, <4 x i32> zeroinitializer
  %r = fmul <4 x float> %v, %sp
  ret <4 x float> %r
}'''

CF2_FACT = '''define internal i32 @cf2_fact(i32 %n) noinline {
entry:
  %z = icmp ule i32 %n, 1
  br i1 %z, label %base, label %rec
base:
  ret i32 1
rec:
  %n1 = sub i32 %n, 1
  %f = call i32 @cf2_fact(i32 %n1)
  %r = mul i32 %n, %f
  ret i32 %r
}'''

CF2_FNPTR = '''define internal i32 @cf2_inc(i32 %x) noinline {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define internal i32 @cf2_dbl(i32 %x) noinline {
entry:
  %r = shl i32 %x, 1
  ret i32 %r
}'''


def group_llvm_control_ext():
    """Loops with several exits, early returns, calls returning aggregates
    and vectors, unreachable, assume/expect, sparse and narrow switches,
    threadgroup reductions with barriers in uniform loops, reconvergence
    before SIMD-group functions, and probes for recursion and indirect calls."""
    cases = []

    def multi_exit(i, v):
        a, lim, s = v[0][i], v[1][i] & 255, 0
        for k in range(32):
            if a >> k & 1:
                return k + 1000
            s += k
            if s > lim:
                return s + 2000
        return 7
    cases.append(ir_case(
        'cf_multi_exit_loop', ['u32', 'u32'], 'u32',
        '%lim = and i32 %a1, 255\nbr label %h\n'
        'h:\n%k = phi i32 [ 0, %entry ], [ %k1, %cont ]\n%s = phi i32 [ 0, %entry ], [ %s1, %cont ]\n'
        '%bit = lshr i32 %a0, %k\n%b1 = and i32 %bit, 1\n%set = icmp ne i32 %b1, 0\n'
        'br i1 %set, label %found, label %latch\n'
        'latch:\n%s1 = add i32 %s, %k\n%over = icmp ugt i32 %s1, %lim\nbr i1 %over, label %ovf, label %cont\n'
        'cont:\n%k1 = add i32 %k, 1\n%end = icmp eq i32 %k1, 32\nbr i1 %end, label %none, label %h\n'
        'found:\n%rf = add i32 %k, 1000\nbr label %exit\n'
        'ovf:\n%ro = add i32 %s1, 2000\nbr label %exit\n'
        'none:\nbr label %exit\n'
        'exit:\n%r = phi i32 [ %rf, %found ], [ %ro, %ovf ], [ 7, %none ]',
        multi_exit, gen=[lambda i, r: r.choice([0, 1 << r.randrange(32), r.randrange(1 << 32)]) &
                         ~((1 << r.randrange(0, 12)) - 1) & 0xffffffff, None]))
    c = full_case('cf_return_in_loop', ['u32'], 'u32',
                  lambda i, v: 0xdeadbeef if any(((v[0][i] >> k) & 0xf) == 0xf for k in range(0, 32, 4))
                  else sum((v[0][i] >> k) & 0xf for k in range(0, 32, 4)),
                  '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
                  'br label %h\n'
                  'h:\n%k = phi i32 [ 0, %entry ], [ %k1, %next ]\n%s = phi i32 [ 0, %entry ], [ %s1, %next ]\n'
                  '%sh = lshr i32 %a, %k\n%d = and i32 %sh, 15\n%bad = icmp eq i32 %d, 15\n'
                  'br i1 %bad, label %early, label %next\n'
                  'early:\nret void\n'
                  'next:\n%s1 = add i32 %s, %d\n%k1 = add i32 %k, 4\n%more = icmp ult i32 %k1, 32\n'
                  'br i1 %more, label %h, label %done\n'
                  'done:\n%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                  'store i32 %s1, ptr addrspace(1) %po\nret void',
                  gen=[lambda i, r: r.randrange(1 << 32) & r.choice([0xeeeeeeee, 0xffffffff, 0x77777777])])
    c.out_init = [0xdeadbeef] * N
    cases.append(c)
    small = lambda i, r: r.randrange(0, 100000)
    cases.append(ir_case('cf_call_aggregate_ptr', ['u32', 'u32'], 'u32',
                         '%rem = alloca i32\n%p = call { i32, i32 } @cf2_divmod(i32 %a0, i32 %a1, ptr %rem)\n'
                         '%q = extractvalue { i32, i32 } %p, 0\n%m = extractvalue { i32, i32 } %p, 1\n'
                         '%m2 = load i32, ptr %rem\n%qm = mul i32 %q, 1000\n%x = add i32 %qm, %m\n'
                         '%r = add i32 %x, %m2', lambda i, v: u32(v[0][i] // v[1][i] * 1000 + 2 * (v[0][i] % v[1][i])),
                         {CF2_HELPERS}, [small, lambda i, r: r.randrange(1, 300)]))
    cases.append(ir_case('cf_call_vector', ['f32'] * 5, 'f32',
                         VEC4_F32 + '%w = call <4 x float> @cf2_scale(<4 x float> %v, float %a4)\n'
                         '%r = extractelement <4 x float> %w, i32 %k',
                         lambda i, v: f32(v[i % 4][i] * v[4][i]), {CF2_VEC}, ulps=0))
    cases.append(ir_case('cf_switch_unreachable', ['u32'], 'u32',
                         '%s = and i32 %a0, 3\n'
                         'switch i32 %s, label %u [ i32 0, label %c0 i32 1, label %c1 i32 2, label %c2 i32 3, label %c3 ]\n'
                         'u:\nunreachable\n'
                         'c0:\nbr label %j\nc1:\n%v1 = add i32 %a0, 5\nbr label %j\n'
                         'c2:\n%v2 = lshr i32 %a0, 3\nbr label %j\nc3:\n%v3 = mul i32 %a0, 7\nbr label %j\n'
                         'j:\n%r = phi i32 [ 99, %c0 ], [ %v1, %c1 ], [ %v2, %c2 ], [ %v3, %c3 ]',
                         lambda i, v: (99, u32(v[0][i] + 5), v[0][i] >> 3, u32(v[0][i] * 7))[v[0][i] & 3]))
    cases.append(ir_case('cf_assume_expect', ['u32'], 'u32',
                         '%m = and i32 %a0, 15\n%in = icmp ult i32 %m, 16\ncall void @llvm.assume(i1 %in)\n'
                         '%big = icmp ugt i32 %m, 12\n%e = call i1 @llvm.expect.i1(i1 %big, i1 false)\n'
                         'br i1 %e, label %rare, label %common\n'
                         'rare:\n%x = mul i32 %m, 100\nbr label %j\ncommon:\n%y = add i32 %m, 1\nbr label %j\n'
                         'j:\n%r = phi i32 [ %x, %rare ], [ %y, %common ]',
                         lambda i, v: (v[0][i] & 15) * 100 if v[0][i] & 15 > 12 else (v[0][i] & 15) + 1,
                         {'declare void @llvm.assume(i1)', 'declare i1 @llvm.expect.i1(i1, i1)'}))
    sparse = [0, 1 << 40, -1, 12345678901, -(1 << 62), 7]
    cases.append(ir_case('cf_switch_i64_sparse', ['i64'], 'u32',
                         'switch i64 %a0, label %d [ i64 0, label %c0 i64 1099511627776, label %c1 '
                         'i64 -1, label %c2 i64 12345678901, label %c3 i64 -4611686018427387904, label %c4 ]\n'
                         'c0:\nbr label %j\nc1:\nbr label %j\nc2:\nbr label %j\nc3:\nbr label %j\nc4:\nbr label %j\n'
                         'd:\nbr label %j\n'
                         'j:\n%r = phi i32 [ 10, %c0 ], [ 11, %c1 ], [ 12, %c2 ], [ 13, %c3 ], [ 14, %c4 ], [ 0, %d ]',
                         lambda i, v: {0: 10, 1 << 40: 11, -1: 12, 12345678901: 13, -(1 << 62): 14}.get(v[0][i], 0),
                         gen=[lambda i, r: r.choice(sparse + [r.randrange(-(1 << 63), 1 << 63)])]))
    cases.append(ir_case('cf_switch_i8', ['i8'], 'i32',
                         'switch i8 %a0, label %d [ i8 -128, label %c0 i8 -1, label %c1 i8 0, label %c2 i8 127, label %c3 ]\n'
                         'c0:\nbr label %j\nc1:\nbr label %j\nc2:\nbr label %j\nc3:\nbr label %j\n'
                         'd:\n%x = sext i8 %a0 to i32\nbr label %j\n'
                         'j:\n%r = phi i32 [ 1000, %c0 ], [ 1001, %c1 ], [ 1002, %c2 ], [ 1003, %c3 ], [ %x, %d ]',
                         lambda i, v: {-128: 1000, -1: 1001, 0: 1002, 127: 1003}.get(v[0][i], v[0][i]),
                         gen=[lambda i, r: r.choice([-128, -1, 0, 127, r.randrange(-128, 128)])]))
    cases.append(ir_case('cf_loop_i64_countdown', ['u32'], 'u64',
                         '%n0 = and i32 %a0, 63\n%n = zext i32 %n0 to i64\nbr label %h\n'
                         'h:\n%k = phi i64 [ %n, %entry ], [ %k1, %b ]\n%s = phi i64 [ 1, %entry ], [ %s1, %b ]\n'
                         '%go = icmp sgt i64 %k, 0\nbr i1 %go, label %b, label %x\n'
                         'b:\n%s3 = mul i64 %s, 3\n%s1 = add i64 %s3, %k\n%k1 = sub i64 %k, 1\nbr label %h\n'
                         'x:\n%r = add i64 %s, 0',
                         lambda i, v: functools.reduce(lambda s, k: u64(s * 3 + k), range(v[0][i] & 63, 0, -1), 1)))
    cases.append(ir_case('cf_nested_if_float_phi', ['f32', 'f32'], 'f32',
                         '%c1 = fcmp ogt float %a0, 0.0\nbr i1 %c1, label %p, label %n\n'
                         'p:\n%c2 = fcmp olt float %a1, %a0\nbr i1 %c2, label %pp, label %pn\n'
                         'pp:\n%x1 = fsub float %a0, %a1\nbr label %j\n'
                         'pn:\n%x2 = fmul float %a1, 2.0\nbr label %j\n'
                         'n:\n%x3 = fneg float %a0\nbr label %j\n'
                         'j:\n%r = phi float [ %x1, %pp ], [ %x2, %pn ], [ %x3, %n ]',
                         lambda i, v: (f32(v[0][i] - v[1][i]) if v[1][i] < v[0][i] else f32(v[1][i] * 2))
                         if v[0][i] > 0 else -v[0][i], ulps=0))
    # A loop whose trip count differs per lane, then a SIMD-group sum: every
    # lane must be back (reconverged) for the reduction.
    cases.append(ir_case('cf_divergent_loop_then_simd_sum', ['u32'], 'u32',
                         '%n = and i32 %a0, 31\nbr label %h\n'
                         'h:\n%k = phi i32 [ 0, %entry ], [ %k1, %h ]\n%s = phi i32 [ 0, %entry ], [ %s1, %h ]\n'
                         '%s1 = add i32 %s, %k\n%k1 = add i32 %k, 1\n%more = icmp ult i32 %k, %n\n'
                         'br i1 %more, label %h, label %x\n'
                         'x:\n%r = call i32 @air.simd_sum.u.i32(i32 %s1)',
                         lambda i, v: u32(sum(sum(range((x & 31) + 1)) for x in lanes(i, v[0]))),
                         {'declare i32 @air.simd_sum.u.i32(i32)'}))
    # Pointer select between two private variables.
    cases.append(ir_case('cf_select_private_ptr', ['u32', 'u32'], 'u32',
                         '%x = alloca i32\n%y = alloca i32\nstore i32 %a0, ptr %x\nstore i32 %a1, ptr %y\n'
                         '%c = icmp ugt i32 %a0, %a1\n%p = select i1 %c, ptr %x, ptr %y\nstore i32 0, ptr %p\n'
                         '%lx = load i32, ptr %x\n%ly = load i32, ptr %y\n%l2 = shl i32 %ly, 1\n%r = add i32 %lx, %l2',
                         lambda i, v: u32((0 if v[0][i] > v[1][i] else v[0][i]) +
                                          2 * (v[1][i] if v[0][i] > v[1][i] else 0))))
    # Threadgroup tree reduction: a barrier inside a loop all threads run.
    c = Case('tg_tree_reduce', ['i32'], 'i32', None, TG_DECLS, lambda i, v: s32(sum(v[0])), tgsize=64)
    c.full = tg_kernel(c.name, (
        '%sp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %lid\n'
        'store i32 %a0, ptr addrspace(3) %sp\ncall void @air.wg.barrier(i32 2, i32 1)\nbr label %h\n'
        'h:\n%s = phi i32 [ 32, %entry ], [ %s1, %latch ]\n%act = icmp ult i32 %lid, %s\n'
        'br i1 %act, label %add, label %latch\n'
        'add:\n%oi = add i32 %lid, %s\n%op = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %oi\n'
        '%x = load i32, ptr addrspace(3) %sp\n%y = load i32, ptr addrspace(3) %op\n%xy = add i32 %x, %y\n'
        'store i32 %xy, ptr addrspace(3) %sp\nbr label %latch\n'
        'latch:\ncall void @air.wg.barrier(i32 2, i32 1)\n%s1 = lshr i32 %s, 1\n%more = icmp ne i32 %s1, 0\n'
        'br i1 %more, label %h, label %done\n'
        'done:\n%r = load i32, ptr addrspace(3) @tg\nstore i32 %r, ptr addrspace(1) %po'))
    cases.append(c)
    # Barriers on both sides of a branch whose condition is uniform.
    def uniform_ref(i, v):
        return v[0][63 - i] if v[0][0] < 0 else v[0][(i + 1) % 64]
    c = Case('tg_uniform_branch_barrier', ['i32'], 'i32', None, TG_DECLS, uniform_ref, tgsize=64)
    c.full = tg_kernel(c.name, (
        '%sp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %lid\n'
        'store i32 %a0, ptr addrspace(3) %sp\n%u = load i32, ptr addrspace(1) %in0\n'
        '%neg = icmp slt i32 %u, 0\nbr i1 %neg, label %rev, label %rot\n'
        'rev:\ncall void @air.wg.barrier(i32 2, i32 1)\n%rl = sub i32 63, %lid\nbr label %j\n'
        'rot:\ncall void @air.wg.barrier(i32 2, i32 1)\n%l1 = add i32 %lid, 1\n%l2 = and i32 %l1, 63\nbr label %j\n'
        'j:\n%idx = phi i32 [ %rl, %rev ], [ %l2, %rot ]\n'
        '%rp = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %idx\n'
        '%r = load i32, ptr addrspace(3) %rp\nstore i32 %r, ptr addrspace(1) %po'))
    cases.append(c)
    # Probe: an indirect call (Metal has no function pointers outside
    # visible functions); expected to be refused.
    # (Recursion is not probed: llc -mtriple=air does not terminate on a
    # recursive internal function and is killed for lack of memory.)
    c = ir_case('cf_indirect_call', ['u32'], 'u32',
                '%odd = trunc i32 %a0 to i1\n%fp = select i1 %odd, ptr @cf2_inc, ptr @cf2_dbl\n'
                '%r = call i32 %fp(i32 %a0)',
                lambda i, v: u32(v[0][i] + 1) if v[0][i] & 1 else u32(v[0][i] * 2), {CF2_FNPTR})
    c.absent = True
    cases.append(c)
    return cases


def group_llvm_memory_ext():
    """Constant tables of floats, structs and 2D arrays; threadgroup arrays
    of structs, bytes, halfwords and vectors; more atomics; volatile;
    memcpy of dynamic length and memmove; private vectors and structs with
    dynamic indices; bool arrays; negative and 64-bit GEP indices; whole
    aggregate loads and stores."""
    cases = []
    coef = [0.5, -1.25, 2.0, 0.375, -3.0, 1.0, 0.0625, -0.75]
    coef = [f32(x) for x in coef]

    def horner(i, v):
        acc, x = 0.0, v[0][i]
        for k in range(8):
            acc = f32(f32(acc * x) + coef[k])
        return acc
    c = full_case('const_float_table_horner', ['f32'], 'f32', horner,
                  '%pa = getelementptr float, ptr addrspace(1) %in0, i32 %tid\n%x = load float, ptr addrspace(1) %pa\n'
                  'br label %h\n'
                  'h:\n%k = phi i32 [ 0, %entry ], [ %k1, %h ]\n%acc = phi float [ 0.0, %entry ], [ %a1, %h ]\n'
                  '%cp = getelementptr [8 x float], ptr addrspace(2) @coef_table, i32 0, i32 %k\n'
                  '%cf = load float, ptr addrspace(2) %cp\n%m = fmul float %acc, %x\n%a1 = fadd float %m, %cf\n'
                  '%k1 = add i32 %k, 1\n%more = icmp ult i32 %k1, 8\nbr i1 %more, label %h, label %x2\n'
                  'x2:\n%po = getelementptr float, ptr addrspace(1) %out, i32 %tid\n'
                  'store float %a1, ptr addrspace(1) %po\nret void', gen=[lambda i, r: r.uniform(-2, 2)], ulps=0)
    c.full = ('@coef_table = internal addrspace(2) constant [8 x float] [%s]\n' %
              ', '.join('float %r' % x for x in coef)) + c.full
    cases.append(c)
    st = [(3, 0.5), (-7, 1.25), (100, -2.0), (0, 8.0)]
    c = full_case('const_struct_table', ['u32'], 'f32', lambda i, v: f32(st[v[0][i] & 3][0] + st[v[0][i] & 3][1]),
                  '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
                  '%k = and i32 %a, 3\n'
                  '%p0 = getelementptr [4 x { i32, float }], ptr addrspace(2) @st_table, i32 0, i32 %k, i32 0\n'
                  '%p1 = getelementptr [4 x { i32, float }], ptr addrspace(2) @st_table, i32 0, i32 %k, i32 1\n'
                  '%i = load i32, ptr addrspace(2) %p0\n%f = load float, ptr addrspace(2) %p1\n'
                  '%fi = sitofp i32 %i to float\n%r = fadd float %fi, %f\n'
                  '%po = getelementptr float, ptr addrspace(1) %out, i32 %tid\nstore float %r, ptr addrspace(1) %po\n'
                  'ret void', ulps=0)
    c.full = ('@st_table = internal addrspace(2) constant [4 x { i32, float }] [%s]\n' %
              ', '.join('{ i32, float } { i32 %d, float %r }' % x for x in st)) + c.full
    cases.append(c)
    t2 = [[r * 4 + c_ for c_ in range(4)] for r in range(4)]
    t2 = [[x * x * 17 % 251 for x in row] for row in t2]
    c = full_case('const_2d_table', ['u32'], 'u32', lambda i, v: t2[v[0][i] >> 2 & 3][v[0][i] & 3],
                  '%pa = getelementptr i32, ptr addrspace(1) %in0, i32 %tid\n%a = load i32, ptr addrspace(1) %pa\n'
                  '%c = and i32 %a, 3\n%r0 = lshr i32 %a, 2\n%rw = and i32 %r0, 3\n'
                  '%p = getelementptr [4 x [4 x i32]], ptr addrspace(2) @t2_table, i32 0, i32 %rw, i32 %c\n'
                  '%r = load i32, ptr addrspace(2) %p\n'
                  '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\nstore i32 %r, ptr addrspace(1) %po\n'
                  'ret void')
    c.full = ('@t2_table = internal addrspace(2) constant [4 x [4 x i32]] [%s]\n' %
              ', '.join('[4 x i32] [%s]' % ', '.join('i32 %d' % x for x in row) for row in t2)) + c.full
    cases.append(c)

    def tg_full(name, ty, body):
        """A kernel over in0/out with threadgroup global @tg_<name> of type ty."""
        k = tg_kernel(name, 'BODY').split('\n', 1)[1]
        return ('@tg_%s = internal addrspace(3) global %s undef, align 16\n' % (name, ty) +
                k.replace('BODY', body.strip().replace('\n', '\n  ').replace('@tg', '@tg_' + name)))
    # Threadgroup array of structs: thread l stores {l*3, a}, reads lane l^1's.
    c = Case('tg_struct_array', ['i32'], 'i32', None, TG_DECLS,
             lambda i, v: s32((i ^ 1) * 3 + v[0][i ^ 1]), tgsize=64,
             gen=[lambda i, r: r.randrange(-1000000, 1000000)])
    c.full = tg_full(c.name, '[64 x { i32, float }]', (
        '%f0 = getelementptr [64 x { i32, float }], ptr addrspace(3) @tg, i32 0, i32 %lid, i32 0\n'
        '%f1 = getelementptr [64 x { i32, float }], ptr addrspace(3) @tg, i32 0, i32 %lid, i32 1\n'
        '%l3 = mul i32 %lid, 3\nstore i32 %l3, ptr addrspace(3) %f0\n%af = sitofp i32 %a0 to float\n'
        'store float %af, ptr addrspace(3) %f1\ncall void @air.wg.barrier(i32 2, i32 1)\n'
        '%o = xor i32 %lid, 1\n'
        '%g0 = getelementptr [64 x { i32, float }], ptr addrspace(3) @tg, i32 0, i32 %o, i32 0\n'
        '%g1 = getelementptr [64 x { i32, float }], ptr addrspace(3) @tg, i32 0, i32 %o, i32 1\n'
        '%x = load i32, ptr addrspace(3) %g0\n%y = load float, ptr addrspace(3) %g1\n'
        '%yi = fptosi float %y to i32\n%r = add i32 %x, %yi\nstore i32 %r, ptr addrspace(1) %po'))
    cases.append(c)
    for bits in (8, 16):
        it = 'i%d' % bits
        c = Case('tg_%s_array' % it, ['u32'], 'u32', None, TG_DECLS,
                 lambda i, v, b=bits: v[0][63 - i] & ((1 << b) - 1), tgsize=64)
        c.full = tg_full(c.name, '[64 x %s]' % it, (
            '%%sp = getelementptr [64 x {t}], ptr addrspace(3) @tg, i32 0, i32 %%lid\n'
            '%%v = trunc i32 %%a0 to {t}\nstore {t} %%v, ptr addrspace(3) %%sp\n'
            'call void @air.wg.barrier(i32 2, i32 1)\n%%rl = sub i32 63, %%lid\n'
            '%%rp = getelementptr [64 x {t}], ptr addrspace(3) @tg, i32 0, i32 %%rl\n'
            '%%x = load {t}, ptr addrspace(3) %%rp\n%%r = zext {t} %%x to i32\n'
            'store i32 %%r, ptr addrspace(1) %%po').replace('%%', '%').format(t=it))
        cases.append(c)
    c = Case('tg_v4f32_lanes', ['f32'], 'f32', None, TG_DECLS,
             lambda i, v: v[0][(15 - i // 4) * 4 + i % 4], tgsize=64, ulps=0)
    c.full = tg_full(c.name, '[16 x <4 x float>]', (
        '%q = lshr i32 %lid, 2\n%k = and i32 %lid, 3\n'
        '%sp = getelementptr [16 x <4 x float>], ptr addrspace(3) @tg, i32 0, i32 %q, i32 %k\n'
        '%af = bitcast i32 %a0 to float\nstore float %af, ptr addrspace(3) %sp\n'
        'call void @air.wg.barrier(i32 2, i32 1)\n%rq = sub i32 15, %q\n'
        '%vp = getelementptr [16 x <4 x float>], ptr addrspace(3) @tg, i32 0, i32 %rq\n'
        '%vv = load <4 x float>, ptr addrspace(3) %vp\n%e = extractelement <4 x float> %vv, i32 %k\n'
        '%ei = bitcast float %e to i32\nstore i32 %ei, ptr addrspace(1) %po'))
    cases.append(c)
    # Unique old values from a threadgroup counter: out[old] = 1 for every thread.
    c = Case('tg_atomic_inc_unique', ['u32'], 'u32', None, TG_DECLS, lambda i, v: 1, tgsize=64,
             out_init=[0] * N)
    c.full = tg_full(c.name, 'i32', (
        '%z = icmp eq i32 %lid, 0\nbr i1 %z, label %init, label %go\n'
        'init:\nstore i32 0, ptr addrspace(3) @tg\nbr label %go\n'
        'go:\ncall void @air.wg.barrier(i32 2, i32 1)\n'
        '%old = atomicrmw add ptr addrspace(3) @tg, i32 1 monotonic\n'
        '%slot = getelementptr i32, ptr addrspace(1) %out, i32 %old\nstore i32 1, ptr addrspace(1) %slot'))
    cases.append(c)
    import operator
    cases.append(atomic_case('atomic_fmax_f32', 'fmax', 'f32', max, init=-1000.0))
    cases[-1].gen = [lambda i, r: float(r.randrange(-100, 100))]
    cases.append(atomic_case('atomic_fmin_f32', 'fmin', 'f32', min, init=1000.0))
    cases[-1].gen = [lambda i, r: float(r.randrange(-100, 100))]
    for name, op, fn, init in (('atomic_add_i64', 'add', operator.add, 0),
                               ('atomic_umax_u64', 'umax', max, 0),
                               ('atomic_umin_u64', 'umin', min, (1 << 64) - 1)):
        c = atomic_case(name, op, 'u64', fn, init=init)
        c.ref = (lambda i, v, fn=fn, init=init: fold_slots(v[0], fn, init, u64)[i])
        cases.append(c)
    c = atomic_case('tg_atomic_fadd_f32', 'fadd', 'f32', operator.add)
    c.gen = [lambda i, r: float(r.randrange(-100, 100))]
    c.tgsize = 64
    c.ref = lambda i, v: float(sum(v[0]))
    c.decls = TG_DECLS
    c.full = tg_full(c.name, 'float', (
        '%z = icmp eq i32 %lid, 0\nbr i1 %z, label %init, label %go\n'
        'init:\nstore float 0.0, ptr addrspace(3) @tg\nbr label %go\n'
        'go:\ncall void @air.wg.barrier(i32 2, i32 1)\n%af = bitcast i32 %a0 to float\n'
        '%old = atomicrmw fadd ptr addrspace(3) @tg, float %af monotonic\n'
        'call void @air.wg.barrier(i32 2, i32 1)\n%r = load float, ptr addrspace(3) @tg\n'
        '%ri = bitcast float %r to i32\nstore i32 %ri, ptr addrspace(1) %po'))
    c.out_init = None
    cases.append(c)
    # A float stored through a pointer computed as an i32 element (and the
    # reverse): opaque pointers carry no element type.
    cases.append(full_case('pun_store_float_i32_gep', ['f32'], 'f32', lambda i, v: f32(v[0][i] + 1),
                           '%pa = getelementptr float, ptr addrspace(1) %in0, i32 %tid\n'
                           '%a = load float, ptr addrspace(1) %pa\n%r = fadd float %a, 1.0\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store float %r, ptr addrspace(1) %po\nret void', ulps=0))
    cases.append(full_case('pun_load_i32_float_gep', ['u32'], 'u32', lambda i, v: v[0][i] ^ 1,
                           '%pa = getelementptr float, ptr addrspace(1) %in0, i32 %tid\n'
                           '%a = load i32, ptr addrspace(1) %pa\n%r = xor i32 %a, 1\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store i32 %r, ptr addrspace(1) %po\nret void'))
    # A compare-exchange loop (how front ends write an atomic float multiply).
    c = full_case('cmpxchg_loop_fmul', ['f32'], 'f32',
                  lambda i, v: functools.reduce(lambda a, b: a * b, v[0][i % 8::8], 1.0) if i < 8 else 1.0,
                  '%pa = getelementptr float, ptr addrspace(1) %in0, i32 %tid\n%a = load float, ptr addrspace(1) %pa\n'
                  '%slot = and i32 %tid, 7\n%sp = getelementptr float, ptr addrspace(1) %out, i32 %slot\n'
                  '%init = load i32, ptr addrspace(1) %sp\nbr label %loop\n'
                  'loop:\n%cur = phi i32 [ %init, %entry ], [ %seen, %loop ]\n'
                  '%cf = bitcast i32 %cur to float\n%nf = fmul float %cf, %a\n%ni = bitcast float %nf to i32\n'
                  '%pair = cmpxchg ptr addrspace(1) %sp, i32 %cur, i32 %ni monotonic monotonic\n'
                  '%seen = extractvalue { i32, i1 } %pair, 0\n%ok = extractvalue { i32, i1 } %pair, 1\n'
                  'br i1 %ok, label %done, label %loop\ndone:\nret void',
                  gen=[lambda i, r: r.choice([0.5, 2.0, -1.0, 1.0, 4.0, -0.25])], ulps=0)
    c.out_init = [1.0] * N
    cases.append(c)
    cases.append(ir_case('volatile_load_store', ['u32'], 'u32',
                         '%v = load volatile i32, ptr addrspace(1) %p0\n%r = xor i32 %v, 85',
                         lambda i, v: v[0][i] ^ 85))
    # memcpy of a dynamic length: thread s < 16 copies (in1[s] & 15) + 1 bytes of its 16-byte slot.
    def memcpy_ref(i, v):
        s, n = i // 4, (v[1][i // 4] & 15) + 1
        src = v[0][i].to_bytes(4, 'little')
        return int.from_bytes(bytes(src[j] if 4 * (i % 4) + j < n else 0xaa for j in range(4)), 'little')
    c = full_case('memcpy_dynamic_len', ['u32', 'u32'], 'u32', memcpy_ref,
                  '%go = icmp ult i32 %tid, 16\nbr i1 %go, label %body, label %done\nbody:\n'
                  '%pb = getelementptr i32, ptr addrspace(1) %in1, i32 %tid\n%b = load i32, ptr addrspace(1) %pb\n'
                  '%n0 = and i32 %b, 15\n%n = add i32 %n0, 1\n%n64 = zext i32 %n to i64\n%off = shl i32 %tid, 4\n'
                  '%src = getelementptr i8, ptr addrspace(1) %in0, i32 %off\n'
                  '%dst = getelementptr i8, ptr addrspace(1) %out, i32 %off\n'
                  'call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %n64, i1 false)\n'
                  'br label %done\ndone:\nret void',
                  {'declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)'})
    c.out_init = [0xaaaaaaaa] * N
    cases.append(c)
    cases.append(ir_case('memmove_overlap_private', ['u32'], 'u32',
                         '%arr = alloca [8 x i32], align 4\nbr label %f\n'
                         'f:\n%k = phi i32 [ 0, %entry ], [ %k1, %f ]\n%v = add i32 %a0, %k\n'
                         '%sl = getelementptr [8 x i32], ptr %arr, i32 0, i32 %k\nstore i32 %v, ptr %sl\n'
                         '%k1 = add i32 %k, 1\n%m = icmp ult i32 %k1, 8\nbr i1 %m, label %f, label %mv\n'
                         'mv:\n%d = getelementptr i32, ptr %arr, i32 1\n'
                         'call void @llvm.memmove.p0.p0.i32(ptr align 4 %d, ptr align 4 %arr, i32 28, i1 false)\n'
                         '%s = and i32 %tid, 7\n%sp = getelementptr [8 x i32], ptr %arr, i32 0, i32 %s\n'
                         '%r = load i32, ptr %sp', lambda i, v: u32(v[0][i] + max(0, i % 8 - 1)),
                         {'declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1 immarg)'}))
    c = full_case('memmove_device_overlap', ['u32'], 'u32',
                  lambda i, v: v[0][i] if i % 8 == 0 else v[0][i - 1],
                  '%go = icmp ult i32 %tid, 8\nbr i1 %go, label %body, label %done\nbody:\n'
                  '%b = shl i32 %tid, 3\n%s = getelementptr i32, ptr addrspace(1) %in0, i32 %b\n'
                  '%d = getelementptr i32, ptr addrspace(1) %out, i32 %b\n'
                  'call void @llvm.memcpy.p1.p1.i32(ptr addrspace(1) %d, ptr addrspace(1) %s, i32 32, i1 false)\n'
                  '%d1 = getelementptr i32, ptr addrspace(1) %d, i32 1\n'
                  'call void @llvm.memmove.p1.p1.i32(ptr addrspace(1) %d1, ptr addrspace(1) %d, i32 28, i1 false)\n'
                  'br label %done\ndone:\nret void',
                  {'declare void @llvm.memcpy.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)',
                   'declare void @llvm.memmove.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i1)'})
    cases.append(c)
    cases.append(ir_case('alloca_vector_dynamic', ['f32'] * 4, 'f32',
                         VEC4_F32 + '%a = alloca <4 x float>\nstore <4 x float> %v, ptr %a\n'
                         '%j = xor i32 %k, 2\n%p = getelementptr float, ptr %a, i32 %j\n%r = load float, ptr %p',
                         lambda i, v: v[(i % 4) ^ 2][i], ulps=0))
    cases.append(ir_case('alloca_struct_dynamic', ['u32', 'u32'], 'u32',
                         '%s = alloca { i32, [4 x i32] }\n%h = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 0\n'
                         'store i32 %a1, ptr %h\n%k = and i32 %a0, 3\n'
                         '%e0 = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 1, i32 0\n'
                         'store i32 10, ptr %e0\n%e1 = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 1, i32 1\n'
                         'store i32 11, ptr %e1\n%e2 = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 1, i32 2\n'
                         'store i32 12, ptr %e2\n%e3 = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 1, i32 3\n'
                         'store i32 13, ptr %e3\n'
                         '%ek = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 1, i32 %k\nstore i32 %tid, ptr %ek\n'
                         '%j = xor i32 %k, 1\n%ej = getelementptr { i32, [4 x i32] }, ptr %s, i32 0, i32 1, i32 %j\n'
                         '%x = load i32, ptr %ej\n%y = load i32, ptr %ek\n%hv = load i32, ptr %h\n'
                         '%xy = mul i32 %x, 1000\n%z = add i32 %xy, %y\n%r = xor i32 %z, %hv',
                         lambda i, v: u32((10 + ((v[0][i] & 3) ^ 1)) * 1000 + i) ^ v[1][i]))
    cases.append(ir_case('bool_load_i1', ['i1', 'u32'], 'u32', '%r = select i1 %a0, i32 %a1, i32 7',
                         lambda i, v: v[1][i] if v[0][i] else 7))
    cases.append(ir_case('bool_i8_trunc_store', ['u8', 'u32'], 'u8',
                         '%b = trunc i8 %a0 to i1\n%c = icmp ugt i32 %a1, 2147483648\n%x = xor i1 %b, %c\n'
                         '%r = zext i1 %x to i8', lambda i, v: (v[0][i] & 1) ^ int(v[1][i] > 1 << 31),
                         gen=[lambda i, r: r.randrange(2), None]))
    cases.append(full_case('gep_negative_index', ['u32'], 'u32', lambda i, v: v[0][63 - i],
                           '%end = getelementptr i32, ptr addrspace(1) %in0, i32 63\n%n = sub i32 0, %tid\n'
                           '%p = getelementptr inbounds i32, ptr addrspace(1) %end, i32 %n\n'
                           '%r = load i32, ptr addrspace(1) %p\n'
                           '%po = getelementptr i32, ptr addrspace(1) %out, i32 %tid\n'
                           'store i32 %r, ptr addrspace(1) %po\nret void'))
    cases.append(full_case('gep_i64_index_nuw', ['f32'], 'f32', lambda i, v: f32(v[0][i ^ 1] * 3),
                           '%t64 = zext i32 %tid to i64\n%j = xor i64 %t64, 1\n'
                           '%p = getelementptr inbounds nuw float, ptr addrspace(1) %in0, i64 %j\n'
                           '%x = load float, ptr addrspace(1) %p\n%r = fmul float %x, 3.0\n'
                           '%po = getelementptr inbounds nuw float, ptr addrspace(1) %out, i64 %t64\n'
                           'store float %r, ptr addrspace(1) %po\nret void', ulps=0))
    # Whole-aggregate loads and stores.
    def agg_gen(i, r):
        return r.randrange(1 << 32) if i % 2 == 0 else struct.unpack('<I', struct.pack('<f', r.uniform(-100, 100)))[0]

    def agg_ref(i, v):
        if i % 2 == 0:
            return u32(v[0][i] + 1)
        x = struct.unpack('<f', struct.pack('<I', v[0][i]))[0]
        return struct.unpack('<I', struct.pack('<f', f32(x * 2)))[0]
    cases.append(full_case('aggregate_struct_load_store', ['u32'], 'u32', agg_ref,
                           '%go = icmp ult i32 %tid, 32\nbr i1 %go, label %body, label %done\nbody:\n'
                           '%ps = getelementptr { i32, float }, ptr addrspace(1) %in0, i32 %tid\n'
                           '%s = load { i32, float }, ptr addrspace(1) %ps\n'
                           '%i = extractvalue { i32, float } %s, 0\n%f = extractvalue { i32, float } %s, 1\n'
                           '%i1 = add i32 %i, 1\n%f2 = fmul float %f, 2.0\n'
                           '%t0 = insertvalue { i32, float } poison, i32 %i1, 0\n'
                           '%t1 = insertvalue { i32, float } %t0, float %f2, 1\n'
                           '%po = getelementptr { i32, float }, ptr addrspace(1) %out, i32 %tid\n'
                           'store { i32, float } %t1, ptr addrspace(1) %po\nbr label %done\ndone:\nret void',
                           gen=[agg_gen]))
    cases.append(full_case('aggregate_array_load_store', ['u32'], 'u32', lambda i, v: v[0][i // 4 * 4 + 3 - i % 4],
                           '%go = icmp ult i32 %tid, 16\nbr i1 %go, label %body, label %done\nbody:\n'
                           '%pa = getelementptr [4 x i32], ptr addrspace(1) %in0, i32 %tid\n'
                           '%a = load [4 x i32], ptr addrspace(1) %pa\n' +
                           ''.join('%%e%d = extractvalue [4 x i32] %%a, %d\n' % (k, k) for k in range(4)) +
                           '%b0 = insertvalue [4 x i32] poison, i32 %e3, 0\n%b1 = insertvalue [4 x i32] %b0, i32 %e2, 1\n'
                           '%b2 = insertvalue [4 x i32] %b1, i32 %e1, 2\n%b3 = insertvalue [4 x i32] %b2, i32 %e0, 3\n'
                           '%po = getelementptr [4 x i32], ptr addrspace(1) %out, i32 %tid\n'
                           'store [4 x i32] %b3, ptr addrspace(1) %po\nbr label %done\ndone:\nret void'))
    le = lambda bs: int.from_bytes(bytes(bs), 'little')

    def unaligned_store_ref(i, v):
        buf = bytearray(b'\xcc' * 4 * N)
        for t in range(28):
            buf[1 + 9 * t:9 + 9 * t] = v[0][t].to_bytes(8, 'little')
        return le(buf[4 * i:4 * i + 4])
    c = full_case('store_unaligned_i64', ['u64'], 'u32', unaligned_store_ref,
                  '%go = icmp ult i32 %tid, 28\nbr i1 %go, label %body, label %done\nbody:\n'
                  '%pa = getelementptr i64, ptr addrspace(1) %in0, i32 %tid\n%a = load i64, ptr addrspace(1) %pa\n'
                  '%o9 = mul i32 %tid, 9\n%o = add i32 %o9, 1\n'
                  '%po = getelementptr i8, ptr addrspace(1) %out, i32 %o\n'
                  'store i64 %a, ptr addrspace(1) %po, align 1\nbr label %done\ndone:\nret void')
    c.out_init = [0xcccccccc] * N
    cases.append(c)
    return cases


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def group_llvm_conv_ext():
    """Narrow integers to and from float, the sign operations on float
    encodings (NaN payloads), <2 x half> as i32, doubles a front end writes
    (constants and exact arithmetic), i1 to float, vector and scalar casts."""
    cases = []
    for name, ti, to, inst, ref, gen in (
            ('sitofp_i8_f32', 'i8', 'f32', 'sitofp i8 %a0 to float', float, None),
            ('uitofp_u8_f32', 'u8', 'f32', 'uitofp i8 %a0 to float', float, None),
            ('sitofp_i16_f32', 'i16', 'f32', 'sitofp i16 %a0 to float', float, None),
            ('uitofp_u16_f32', 'u16', 'f32', 'uitofp i16 %a0 to float', float, None),
            ('fptosi_f32_i8', 'f32', 'i8', 'fptosi float %a0 to i8', int, lambda i, r: r.uniform(-128.99, 127.99)),
            ('fptoui_f32_u8', 'f32', 'u8', 'fptoui float %a0 to i8', int, lambda i, r: r.uniform(0, 255.99)),
            ('fptosi_f32_i16', 'f32', 'i16', 'fptosi float %a0 to i16', int, lambda i, r: r.uniform(-32768.9, 32767.9)),
            ('fptoui_f32_u16', 'f32', 'u16', 'fptoui float %a0 to i16', int, lambda i, r: r.uniform(0, 65535.9)),
            ('uitofp_i1_f32', 'u32', 'f32', '%b = trunc i32 %a0 to i1\n%r = uitofp i1 %b to float',
             lambda x: float(x & 1), None),
            ('sitofp_i1_f32', 'u32', 'f32', '%b = trunc i32 %a0 to i1\n%r = sitofp i1 %b to float',
             lambda x: -1.0 if x & 1 else 0.0, None),
            ('trunc_i64_i8', 'i64', 'i8', 'trunc i64 %a0 to i8', s8, None),
            ('sext_i16_i64', 'i16', 'i64', 'sext i16 %a0 to i64', int, None),
            ('zext_i8_i64', 'u8', 'u64', 'zext i8 %a0 to i64', int, None)):
        body = inst if inst.startswith('%') else '%r = ' + inst
        cases.append(ir_case(name, [ti], to, body, lambda i, v, ref=ref: ref(v[0][i]), gen=[gen],
                             ulps=0 if to == 'f32' else None, bitexact=to == 'f32'))
    # Sign operations act on the encoding: NaN payloads (and subnormals) intact.
    nan_bits = lambda i, r: r.choice([0x7f800001 + r.randrange(0x3fffff), 0x7fc00000 + r.randrange(0x3fffff),
                                      0xff800001 + r.randrange(0x7fffff), r.randrange(1 << 32)]) & 0xffffffff
    normal_bits = lambda i, r: r.choice([nan_bits(i, r), r.randrange(0x00800000, 0x7f800000) | r.randrange(2) << 31])
    sub_bits = lambda i, r: r.randrange(1, 0x800000) | r.randrange(2) << 31
    for name, gen in (('nan', normal_bits), ('subnormal', sub_bits)):
        cases.append(ir_case('bits_fneg_' + name, ['u32'], 'u32',
                             '%f = bitcast i32 %a0 to float\n%n = fneg float %f\n%r = bitcast float %n to i32',
                             lambda i, v: v[0][i] ^ 0x80000000, gen=[gen]))
        cases.append(ir_case('bits_fabs_' + name, ['u32'], 'u32',
                             '%f = bitcast i32 %a0 to float\n%n = call float @llvm.fabs.f32(float %f)\n'
                             '%r = bitcast float %n to i32', lambda i, v: v[0][i] & 0x7fffffff,
                             {'declare float @llvm.fabs.f32(float)'}, gen=[gen]))
        cases.append(ir_case('bits_copysign_' + name, ['u32', 'u32'], 'u32',
                             '%f = bitcast i32 %a0 to float\n%s = bitcast i32 %a1 to float\n'
                             '%n = call float @llvm.copysign.f32(float %f, float %s)\n%r = bitcast float %n to i32',
                             lambda i, v: v[0][i] & 0x7fffffff | v[1][i] & 0x80000000,
                             {'declare float @llvm.copysign.f32(float, float)'}, gen=[gen, None]))
    cases.append(ir_case('bits_select_float', ['u32', 'u32'], 'u32',
                         '%f = bitcast i32 %a0 to float\n%g = bitcast i32 %a1 to float\n%c = trunc i32 %tid to i1\n'
                         '%s = select i1 %c, float %f, float %g\n%r = bitcast float %s to i32',
                         lambda i, v: v[0][i] if i & 1 else v[1][i], gen=[nan_bits, sub_bits]))
    c = full_case('bits_float_load_store', ['u32'], 'u32', lambda i, v: v[0][i ^ 1],
                  '%j = xor i32 %tid, 1\n%pa = getelementptr float, ptr addrspace(1) %in0, i32 %j\n'
                  '%f = load float, ptr addrspace(1) %pa\n%po = getelementptr float, ptr addrspace(1) %out, i32 %tid\n'
                  'store float %f, ptr addrspace(1) %po\nret void', gen=[lambda i, r: r.choice([nan_bits(i, r), sub_bits(i, r)])])
    cases.append(c)
    cases.append(ir_case('bits_pun_private', ['f32'], 'u32',
                         '%m = alloca float\n%x = fmul float %a0, 3.0\nstore float %x, ptr %m\n%r = load i32, ptr %m',
                         lambda i, v: f32_bits(f32(v[0][i] * 3))))
    # <2 x half> in an i32.
    h2 = lambda i, r: f16_bits(f16(r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randrange(-10, 10))) | \
        f16_bits(f16(r.choice([-1, 1]) * r.uniform(1, 2) * 2.0 ** r.randrange(-10, 10))) << 16
    cases.append(ir_case('v2f16_bitcast_fmul', ['u32'], 'u32',
                         '%h = bitcast i32 %a0 to <2 x half>\n%m = fmul <2 x half> %h, <half 2.0, half 0.5>\n'
                         '%r = bitcast <2 x half> %m to i32',
                         lambda i, v: f16_bits(f16(bits_f16(v[0][i] & 0xffff) * 2)) |
                         f16_bits(f16(bits_f16(v[0][i] >> 16) * 0.5)) << 16, gen=[h2]))
    cases.append(ir_case('bitcast_v2i32_i64', ['u32', 'u32'], 'u64',
                         '%x0 = insertelement <2 x i32> poison, i32 %a0, i32 0\n'
                         '%x = insertelement <2 x i32> %x0, i32 %a1, i32 1\n%r = bitcast <2 x i32> %x to i64',
                         lambda i, v: v[1][i] << 32 | v[0][i]))
    cases.append(ir_case('bitcast_i64_v2f32', ['u64'], 'f32',
                         '%v = bitcast i64 %a0 to <2 x float>\n%k = and i32 %tid, 1\n'
                         '%r = extractelement <2 x float> %v, i32 %k',
                         lambda i, v: struct.unpack('<f', struct.pack('<I', v[0][i] >> 32 * (i & 1) & 0xffffffff))[0],
                         gen=[lambda i, r: f32_bits(r.uniform(-9, 9)) | f32_bits(r.uniform(-9, 9)) << 32], ulps=0))
    # Doubles as front ends write them (AIR has no double: the backend
    # demotes them); only cases where the double result is exact in float.
    cases.append(ir_case('fptrunc_double_const', ['f32'], 'f32',
                         '%c = fptrunc double 0x3FB999999999999A to float\n%r = fmul float %a0, %c',
                         lambda i, v: f32(v[0][i] * f32(0.1)), ulps=0))
    cases.append(ir_case('double_exact_arith', ['f32'], 'f32',
                         '%d = fpext float %a0 to double\n%m = fmul double %d, 3.0\n%s = fadd double %m, 0.5\n'
                         '%r = fptrunc double %s to float', lambda i, v: v[0][i] * 3 + 0.5,
                         gen=[lambda i, r: float(r.randrange(-100000, 100000))], ulps=0))
    cases.append(ir_case('sitofp_i32_double_fptrunc', ['i32'], 'f32',
                         '%d = sitofp i32 %a0 to double\n%r = fptrunc double %d to float',
                         lambda i, v: int_to_f32(v[0][i]), ulps=0))
    cases.append(ir_case('double_compare', ['f32', 'f32'], 'u32',
                         '%x = fpext float %a0 to double\n%y = fpext float %a1 to double\n'
                         '%c = fcmp olt double %x, %y\n%r = zext i1 %c to i32',
                         lambda i, v: int(v[0][i] < v[1][i])))
    return cases


DISPATCH_ROLES = ['thread_position_in_grid', 'thread_position_in_threadgroup',
                  'threadgroup_position_in_grid', 'threads_per_threadgroup', 'threadgroups_per_grid',
                  'threads_per_grid']


def dispatch_case(name, grid, group, body, decls, ref):
    """A kernel over a grid of 64 threads with shape grid and threadgroups
    group: the thread at (x, y, z) stores %r at out[x + gx*(y + gy*z)]."""
    gx, gy, _ = grid
    c = Case(name, ['u32'], 'u32', None, set(decls), lambda i, v: ref(i % gx, i // gx % gy, i // (gx * gy)))
    c.grid, c.group = list(grid), list(group)
    c.full = ('define void @%s(ptr addrspace(1) %%in0, ptr addrspace(1) %%out) {\nentry:\n'
              '  %%t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()\n'
              '  %%x = extractvalue { i32, i32, i32 } %%t, 0\n  %%y = extractvalue { i32, i32, i32 } %%t, 1\n'
              '  %%z = extractvalue { i32, i32, i32 } %%t, 2\n'
              '  %%zy = mul i32 %%z, %d\n  %%yz = add i32 %%y, %%zy\n  %%row = mul i32 %%yz, %d\n'
              '  %%lin = add i32 %%row, %%x\n  ' % (name, gy, gx) +
              body.strip().replace('\n', '\n  ') +
              '\n  %po = getelementptr i32, ptr addrspace(1) %out, i32 %lin\n'
              '  store i32 %r, ptr addrspace(1) %po\n  ret void\n}')
    return c


def group_dispatch_3d():
    """System values and threadgroup memory in 2D and 3D grids: the y and z
    components, the linear thread index in a 3D threadgroup, the OpenCL-style
    ids with dimension 1 and 2."""
    cases = []
    for tag, grid, group in (('2d', (16, 4, 1), (8, 2, 1)), ('3d', (4, 4, 4), (2, 2, 2))):
        G, T = grid, group
        values = {
            'thread_position_in_grid': lambda p: p,
            'thread_position_in_threadgroup': lambda p, T=T: tuple(p[d] % T[d] for d in range(3)),
            'threadgroup_position_in_grid': lambda p, T=T: tuple(p[d] // T[d] for d in range(3)),
            'threads_per_threadgroup': lambda p, T=T: T,
            'threadgroups_per_grid': lambda p, T=T, G=G: tuple(G[d] // T[d] for d in range(3)),
            'threads_per_grid': lambda p, G=G: G,
        }
        for role in DISPATCH_ROLES:
            for comp in ((1, 2) if tag == '3d' else (0, 1)):
                cases.append(dispatch_case(
                    'dsp%s_%s_%d' % (tag, role, comp), G, T,
                    '%%s = call { i32, i32, i32 } @air.%s.v3i32()\n%%r = extractvalue { i32, i32, i32 } %%s, %d'
                    % (role, comp), {'declare { i32, i32, i32 } @air.%s.v3i32()' % role},
                    lambda x, y, z, f=values[role], comp=comp: f((x, y, z))[comp]))
        lin = lambda x, y, z, T=T: x % T[0] + T[0] * (y % T[1] + T[1] * (z % T[2]))
        for role in ('thread_index_in_threadgroup', 'thread_index_in_simdgroup'):
            cases.append(dispatch_case('dsp%s_%s' % (tag, role), G, T, '%%r = call i32 @air.%s.i32()' % role,
                                       {'declare i32 @air.%s.i32()' % role}, lin))
        for fn, dim, f in (('get_global_id', 1, lambda x, y, z: y), ('get_global_id', 2, lambda x, y, z: z),
                           ('get_local_id', 1, lambda x, y, z, T=T: y % T[1]),
                           ('get_group_id', 1, lambda x, y, z, T=T: y // T[1])):
            if tag == '2d' and dim == 2:
                continue
            cases.append(dispatch_case('dsp%s_air_%s_%d' % (tag, fn, dim), G, T,
                                       '%%r = call i32 @air.%s.i32(i32 %d)' % (fn, dim),
                                       {'declare i32 @air.%s.i32(i32)' % fn}, f))
        # Threadgroup memory indexed by the linear thread index: reverse within the threadgroup.
        size = T[0] * T[1] * T[2]
        c = dispatch_case('dsp%s_tg_reverse' % tag, G, T,
                          '%%li = call i32 @air.thread_index_in_threadgroup.i32()\n'
                          '%%sp = getelementptr [64 x i32], ptr addrspace(3) @tgd%s, i32 0, i32 %%li\n'
                          'store i32 %%lin, ptr addrspace(3) %%sp\ncall void @air.wg.barrier(i32 2, i32 1)\n'
                          '%%rl = sub i32 %d, %%li\n'
                          '%%rp = getelementptr [64 x i32], ptr addrspace(3) @tgd%s, i32 0, i32 %%rl\n'
                          '%%r = load i32, ptr addrspace(3) %%rp' % (tag, size - 1, tag),
                          {'declare i32 @air.thread_index_in_threadgroup.i32()',
                           'declare void @air.wg.barrier(i32, i32)'}, None)

        def rev(x, y, z, T=T, G=G, lin=lin):
            l = lin(x, y, z)
            r = T[0] * T[1] * T[2] - 1 - l
            lx, ly, lz = r % T[0], r // T[0] % T[1], r // (T[0] * T[1])
            bx, by, bz = x - x % T[0], y - y % T[1], z - z % T[2]
            return (bx + lx) + G[0] * ((by + ly) + G[1] * (bz + lz))
        c.ref = lambda i, v, rev=rev, G=G: rev(i % G[0], i // G[0] % G[1], i // (G[0] * G[1]))
        c.full = '@tgd%s = internal addrspace(3) global [64 x i32] undef, align 4\n' % tag + c.full
        cases.append(c)
    return cases


GROUPS = {
    'llvm-float-ieee': group_llvm_float_ieee,
    'llvm-int-ops': group_llvm_int_ops,
    'llvm-vector': group_llvm_vector,
    'llvm-control': group_llvm_control,
    'llvm-conv': group_llvm_conv,
    'llvm-memory-ir': group_llvm_memory_ir,
    'llvm-math-f32': group_llvm_math_f32,
    'air-math-f32': group_air_math_f32,
    'llvm-math-afn': group_llvm_math_afn,
    'llvm-int': group_llvm_int,
    'simd': group_simd,
    'simd-divergent': group_simd_divergent,
    'atomics': group_atomics,
    'threadgroup': group_threadgroup,
    'sysvals': group_sysvals,
    'types': group_types,
    'memory': group_memory,
    'matrix': group_matrix,
    'misc': group_misc,
    'extra': group_extra,
    'llvm-half': group_llvm_half,
    'llvm-int-narrow': group_llvm_int_narrow,
    'llvm-vector-ext': group_llvm_vector_ext,
    'llvm-math-ext': group_llvm_math_ext,
    'llvm-control-ext': group_llvm_control_ext,
    'llvm-memory-ext': group_llvm_memory_ext,
    'llvm-conv-ext': group_llvm_conv_ext,
    'dispatch-3d': group_dispatch_3d,
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
    ap.add_argument('--agx-run', help="run on AGX: the AGX Performance Simulator's agx_run")
    ap.add_argument('--mcpu', default='agx-g14g', help='AGX processor (with --agx-run)')
    ap.add_argument('--objcopy', default='llvm-objcopy')
    ap.add_argument('--readelf', default='llvm-readelf')
    ap.add_argument('--compare-backends', action='store_true',
                    help='run each case on the device and on AGX (needs --agx-run) and compare the outputs with '
                         'each other: SAME or DIFF-<ftz|nan|ulp|real>')
    ap.add_argument('--progress', help='append the name of each request to this file before sending it')
    args = ap.parse_args()
    global PROGRESS
    PROGRESS = args.progress
    load_hangs()
    if args.compare_backends and not args.agx_run:
        ap.error('--compare-backends needs --agx-run')
    agx = args.agx_run and {'run': args.agx_run, 'mcpu': args.mcpu, 'objcopy': args.objcopy,
                            'readelf': args.readelf}

    global MODULE_SUFFIX
    MODULE_SUFFIX = args.module_suffix.replace('\\n', '\n')
    all_results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = args.keep or tmp
        os.makedirs(tmp, exist_ok=True)
        for g in args.group or sorted(GROUPS):
            print('== %s' % g)
            try:
                res = run_group(GROUPS[g](), args.llc, args.queue, tmp, print, agx, g, args.compare_backends)
            except DeviceHang as e:
                print('DEVICE WORK STOPPED: a request timed out (added to %s): %s' % (HANGS_FILE, e))
                if args.json:
                    json.dump(all_results, open(args.json, 'w'), indent=1)
                return 2
            for name, r in res.items():
                if args.compare_backends:
                    print('%-12s %-28s air=%s agx=%s %s' % (r['status'], name, r['air'], r['agx'],
                                                           r.get('detail', '')))
                    continue
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
    if args.compare_backends:
        return 1 if any(k == 'DIFF-real' for k in counts) else 0
    return 1 if set(counts) - {'PASS', 'ABSENT', 'HOST-LIMIT', 'HW-FTZ', 'SKIPPED'} else 0


if __name__ == '__main__':
    sys.exit(main())
