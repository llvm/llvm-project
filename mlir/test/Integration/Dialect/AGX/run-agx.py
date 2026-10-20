#!/usr/bin/env python3
"""Compile an AGX kernel (LLVM IR) and run it on the functional simulator.

Usage:
  run-agx.py --llc LLC --objcopy OBJCOPY --agx-run AGX_RUN
      [--mlir-translate MLIR_TRANSLATE] kernel.{ll,mlir} --grid N --wg N ARG...

A .mlir input is the output of the conversion to the LLVM and AGXIR
dialects: the functions of its gpu.module are translated to LLVM IR.

Each ARG is a kernel argument in order: `TYPE:V,V,...` is a buffer
initialized with the values (TYPE is i32, u32, i64, f16, f32 or f64),
`TYPE:iota*N` one with 0, 1, ..., N - 1, `TYPE:@FILE` one with the values in
FILE, `TYPE*N` an output buffer of N
elements, `inout:TYPE:...` an initialized buffer that is also printed, and
`i32=V` / `i64=V` / `index=V` / `f32=V` a scalar. Arguments are passed as
pointers in uniform registers (see AGXUsage.rst). The output buffers are
printed one per line, `NAME: v v v ...`, with NAME the argument number.
--grid and --wg take X or X,Y or X,Y,Z threads.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

FMT = {'i32': 'i', 'u32': 'I', 'f16': 'e', 'f32': 'f', 'f64': 'd', 'i64': 'q'}


def translate(mlir_translate, path, tmp):
    """The functions of the gpu.module of an MLIR file, as LLVM IR."""
    lines = open(path).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith('gpu.module'))
    depth = lines[start].count('{') - lines[start].count('}')
    end = start
    while depth:
        end += 1
        depth += lines[end].count('{') - lines[end].count('}')
    body = lines[start + 1:end]
    mlir = os.path.join(tmp, 'k.mlir')
    open(mlir, 'w').write('module {\n' + '\n'.join(body) + '\n}\n')
    ll = os.path.join(tmp, 'k.ll')
    subprocess.run([mlir_translate, '-mlir-to-llvmir', mlir, '-o', ll], check=True)
    return ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llc', required=True)
    ap.add_argument('--objcopy', required=True)
    ap.add_argument('--agx-run', dest='agx_run', required=True)
    ap.add_argument('--mlir-translate', dest='mlir_translate')
    ap.add_argument('--cpu', default='agx-g13g')
    ap.add_argument('--grid', required=True)
    ap.add_argument('--wg', required=True)
    ap.add_argument('ir')
    ap.add_argument('args', nargs='*')
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if a.ir.endswith('.mlir'):
            a.ir = translate(a.mlir_translate, a.ir, tmp)
        asm = subprocess.run([a.llc, '-mtriple=agx', '-mcpu=' + a.cpu, a.ir, '-o', '-'],
                             capture_output=True, text=True, check=True).stdout
        obj = os.path.join(tmp, 'k.o')
        subprocess.run([a.llc, '-mtriple=agx', '-mcpu=' + a.cpu, '-filetype=obj', a.ir,
                        '-o', obj], check=True)
        code = os.path.join(tmp, 'k.bin')
        subprocess.run([a.objcopy, '-O', 'binary', '--only-section=.text', obj, code],
                       check=True)
        glob = os.path.join(tmp, 'g.bin')
        subprocess.run([a.objcopy, '-O', 'binary', '--only-section=.agx.globals', obj, glob],
                       check=False)
        grid = int(re.search(r'threadgroups_per_grid at u(\d+)', asm).group(1))
        gm = re.search(r'global data buffer at u(\d+)_u\d+', asm)
        dims = lambda s: (list(map(int, s.split(','))) + [1, 1])[:3]
        grid_d, wg_d = dims(a.grid), dims(a.wg)
        cmd = [a.agx_run, code, '--grid', ','.join(map(str, grid_d)), '--wg', ','.join(map(str, wg_d))]
        uniforms = []
        outs = []
        for i, arg in enumerate(a.args):
            base = 0x100000 * (i + 1)
            if '=' in arg and ':' not in arg:
                # A scalar: 32-bit values take one uniform, 64-bit ones (index
                # is 64-bit) two, low word first.
                t, v = arg.split('=')
                if t == 'f32':
                    uniforms += [struct.unpack('<I', struct.pack('<f', float(v)))[0]]
                    continue
                v = int(v, 0)
                uniforms += [v & 0xffffffff] if t == 'i32' else [v & 0xffffffff, (v >> 32) & 0xffffffff]
                continue
            if '*' in arg and ':' not in arg:
                t, n = arg.split('*')
                size = struct.calcsize('<' + FMT[t]) * int(n)
                f = os.path.join(tmp, 'o%d.bin' % i)
                cmd += ['--alloc', '%#x=%d' % (base, size), '--dump', '%#x=%d:%s' % (base, size, f)]
                outs.append((i, t, int(n), f))
            else:
                inout = arg.startswith('inout:')
                t, vals = arg[6 if inout else 0:].split(':', 1)
                f = os.path.join(tmp, 'i%d.bin' % i)
                if vals.startswith('@'):
                    vals = open(vals[1:]).read().strip()
                if vals.startswith('iota*'):
                    vals = ','.join(str(v) for v in range(int(vals[5:])))
                vs = [float(v) if t[0] == 'f' else int(v, 0) for v in vals.split(',')]
                open(f, 'wb').write(struct.pack('<%d%s' % (len(vs), FMT[t]), *vs))
                cmd += ['--map', '%#x=%s' % (base, f)]
                if inout:
                    fo = os.path.join(tmp, 'o%d.bin' % i)
                    cmd += ['--dump', '%#x=%d:%s' % (base, len(vs) * struct.calcsize('<' + FMT[t]), fo)]
                    outs.append((i, t, len(vs), fo))
            uniforms += [base, 0]
        for k, v in enumerate(uniforms):
            cmd += ['--uniform', '%d=%d' % (k, v)]
        for d in range(3):
            cmd += ['--uniform', '%d=%d' % (grid + d, -(-grid_d[d] // wg_d[d]))]
        if gm:
            g = os.path.join(tmp, 'g2.bin')
            data = open(glob, 'rb').read() if os.path.exists(glob) else b''
            open(g, 'wb').write(data + bytes(16))
            cmd += ['--map', '%#x=%s' % (0x4000000, g), '--uniform', '%s=%d' % (gm.group(1), 0x4000000),
                    '--uniform', '%d=0' % (int(gm.group(1)) + 1)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit('simulator error: ' + r.stdout + r.stderr)
        for i, t, n, f in outs:
            vals = struct.unpack('<%d%s' % (n, FMT[t]), open(f, 'rb').read())
            print('%d: %s' % (i, ' '.join(('%g' % v) if t[0] == 'f' else str(v) for v in vals)))


if __name__ == '__main__':
    main()
