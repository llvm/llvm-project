"""Writes the inputs and the expected result of an M x N x K GEMM test.

usage: gemm_data.py OUTDIR M N K   writes a.txt, b.txt and c.txt
       gemm_data.py --check C.txt  compares run-agx.py output (stdin) with C.txt
A[i][k] = (i * K + k) % 13 and B[k][j] = (k * N + j) % 7, so every product
and sum is an exact integer in f32.
"""
import os
import sys

if sys.argv[1] == '--check':
    want = open(sys.argv[2]).read().split(',')
    got = sys.stdin.read().strip().splitlines()[-1].split(': ', 1)[1].split()
    bad = [i for i in range(len(want)) if float(got[i]) != float(want[i])]
    print('gemm: %d of %d elements wrong%s' % (len(bad), len(want),
          '' if not bad else ', first at %d: %s != %s' % (bad[0], got[bad[0]], want[bad[0]])))
    sys.exit(1 if bad else 0)

out, M, N, K = sys.argv[1], *map(int, sys.argv[2:5])
os.makedirs(out, exist_ok=True)
a = [(i * K + k) % 13 for i in range(M) for k in range(K)]
b = [(k * N + j) % 7 for k in range(K) for j in range(N)]
c = [sum(a[i * K + k] * b[k * N + j] for k in range(K)) for i in range(M) for j in range(N)]
for name, v in (('a', a), ('b', b), ('c', c)):
    open(os.path.join(out, name + '.txt'), 'w').write(','.join(map(str, v)))
