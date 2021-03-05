# RUN: %PYTHON -m mlir.tools.linalg_opdsl.dump_oplib --file %s | FileCheck %s

from mlir.tools.linalg_opdsl.lang import *


# Verify that simple case with iteration order defined lexically and reduction
# dims auto discovered emits the right shape, indexing maps and iterator types.
# CHECK: ---
# CHECK-LABEL: matmul
# CHECK: shape: affine_map<()[s0, s1, s2] -> (s0, s2)>
# CHECK: shape: affine_map<()[s0, s1, s2] -> (s2, s1)>
# CHECK: shape: affine_map<()[s0, s1, s2] -> (s0, s1)>
# CHECK: static_indexing_maps:
# CHECK-NEXT: - affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>
# CHECK-NEXT: - affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>
# CHECK-NEXT: - affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>
# CHECK: iterator_types:
# CHECK-NEXT: - parallel
# CHECK-NEXT: - parallel
# CHECK-NEXT: - reduction
@linalg_structured_op
def matmul(A=TensorDef(T, S.M, S.K),
           B=TensorDef(T, S.K, S.N),
           C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


# Verifies that assignment to a scalar (represented as [None]) is represented
# correctly.
# CHECK: ---
# CHECK-LABEL: dot
# CHECK: shape: affine_map<()[s0] -> (s0)>
# CHECK: shape: affine_map<()[s0] -> (s0)>
# CHECK: shape: affine_map<()[s0] -> ()>
# CHECK: static_indexing_maps:
# CHECK-NEXT: - affine_map<(d0)[s0] -> (d0)>
# CHECK-NEXT: - affine_map<(d0)[s0] -> (d0)>
# CHECK-NEXT: - affine_map<(d0)[s0] -> ()>
# CHECK: iterator_types:
# CHECK-NEXT: - reduction
@linalg_structured_op
def dot(A=TensorDef(T, S.M), B=TensorDef(T, S.M), C=TensorDef(U, output=True)):
  C[None] += cast(U, A[D.m]) * cast(U, B[D.m])
