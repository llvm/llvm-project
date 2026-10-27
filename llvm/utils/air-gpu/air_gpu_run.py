#!/usr/bin/env python3
"""Run an AIR library on the /air service and check readback against expected values.

Designed to be callable from lit:

  RUN: llc -mtriple=air -filetype=obj %s -o %t.metallib
  RUN: %air_gpu_run %t.metallib --kernel probe --grid 4 --group 4 \
  RUN:     --buffer 16 --expect-u32 7,8,9,10

Every AIR library handed to this script must come from our own backend; the
service refuses source compilation, so there is no path by which Apple's
toolchain output could enter a test.
"""
import argparse
import base64
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, '/air/client')
from client import submit  # noqa: E402

FORMATS = {'u32': ('<%dI', 4), 'i32': ('<%di', 4), 'f32': ('<%df', 4)}


def call(request, queue):
    reply = submit(request, queue)
    if not reply.get('ok'):
        raise SystemExit(f'AIR service rejected the request:\n{json.dumps(reply, indent=2)}')
    return reply['result']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('air', type=Path)
    ap.add_argument('--kernel', default='probe')
    ap.add_argument('--grid', default='4', help='X[,Y,Z]')
    ap.add_argument('--group', default='4', help='X[,Y,Z]')
    ap.add_argument('--buffer', action='append', default=[], type=int,
                    help='byte size of a buffer, repeatable; bound at its index')
    ap.add_argument('--upload-u32', action='append', default=[],
                    help='IDX=v0,v1,... initial uint32 contents for a buffer')
    ap.add_argument('--expect-u32', help='comma-separated expected uint32 of buffer 0')
    ap.add_argument('--expect-f32', help='comma-separated expected float32 of buffer 0')
    ap.add_argument('--tolerance', type=float, default=0.0)
    ap.add_argument('--dispatches', type=int, default=1,
                    help='repeat the command N times, in order')
    ap.add_argument('--expect-functions', help='comma-separated names inspect must report')
    ap.add_argument('--queue', default='/air')
    args = ap.parse_args()

    def dims(text):
        parts = [int(v) for v in text.split(',')]
        return (parts + [1, 1])[:3]

    encoded = base64.b64encode(args.air.read_bytes()).decode()

    functions = call({'op': 'inspect', 'air': encoded}, args.queue)['functions']
    if args.expect_functions:
        missing = [n for n in args.expect_functions.split(',') if n not in functions]
        if missing:
            raise SystemExit(f'inspect did not report {missing}; got {functions}')
    if args.kernel not in functions:
        raise SystemExit(f'kernel {args.kernel!r} absent from inspected AIR; got {functions}')

    sizes = args.buffer or [16]
    buffers = [{'size': size} for size in sizes]
    for spec in args.upload_u32:
        index, _, values = spec.partition('=')
        data = [int(v) for v in values.split(',')]
        buffers[int(index)]['data'] = base64.b64encode(
            struct.pack('<%dI' % len(data), *data)).decode()

    command = {'kernel': args.kernel,
               'bindings': [{'index': i, 'buffer': i} for i in range(len(buffers))],
               'grid': dims(args.grid), 'group': dims(args.group)}
    job = {'op': 'execute', 'air': encoded, 'buffers': buffers,
           'commands': [dict(command) for _ in range(args.dispatches)],
           'readback': [0]}

    result = call(job, args.queue)
    raw = base64.b64decode(result['buffers'][0]['data'])

    if args.expect_u32:
        want = tuple(int(v) for v in args.expect_u32.split(','))
        got = struct.unpack('<%dI' % len(want), raw[:4 * len(want)])
        if got != want:
            raise SystemExit(f'readback mismatch: got {got}, want {want}')
    elif args.expect_f32:
        want = tuple(float(v) for v in args.expect_f32.split(','))
        got = struct.unpack('<%df' % len(want), raw[:4 * len(want)])
        if any(abs(g - w) > args.tolerance for g, w in zip(got, want)):
            raise SystemExit(f'readback mismatch: got {got}, want {want}')
    else:
        raise SystemExit('nothing to check: pass --expect-u32 or --expect-f32')

    if len(result.get('dispatches', [])) != args.dispatches:
        raise SystemExit(f'expected {args.dispatches} dispatches, '
                         f'got {len(result.get("dispatches", []))}')

    print(f'OK {args.kernel} on {result["device"]}')


if __name__ == '__main__':
    main()
