# RUN: %PYTHON -m mlir.dialects.linalg.opdsl.dump_oplib --file %s | FileCheck %s

from mlir.dialects.linalg.opdsl.lang import *


# CHECK: ---
# CHECK-LABEL: matmul
# CHECK: args:
# CHECK:     name: A
# CHECK:     usage: input
# CHECK:     shape: affine_map<()[s0, s1, s2] -> (s0, s2)>
# CHECK:     element_type_var: T
# CHECK:     name: B
# CHECK:     usage: input
# CHECK:     shape: affine_map<()[s0, s1, s2] -> (s2, s1)>
# CHECK:     element_type_var: T
# CHECK:     name: C
# CHECK:     usage: output
# CHECK:     shape: affine_map<()[s0, s1, s2] -> (s0, s1)>
# CHECK:     element_type_var: U
@linalg_structured_op
def matmul(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


# CHECK: ---
# CHECK-LABEL: fill
# CHECK: captures:
# CHECK: - !<LinalgCaptureDef>
# CHECK:   name: value
# CHECK:   type_var: T
@linalg_structured_op
def fill(O=TensorDef(T, S.M, S.K, output=True), value=CaptureDef(T)):
  O[D.m, D.n] = value
