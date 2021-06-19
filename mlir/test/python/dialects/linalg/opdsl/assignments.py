# RUN: %PYTHON -m mlir.dialects.linalg.opdsl.dump_oplib --file %s | FileCheck %s

from mlir.dialects.linalg.opdsl.lang import *


# CHECK: ---
# CHECK-LABEL: matmul
# CHECK: assignments:
# CHECK:  -
# CHECK:    arg: C
# CHECK:    value:
# CHECK:      scalar_apply:
# CHECK:        fn_name: add
# CHECK:        operands:
# CHECK:          scalar_apply:
# CHECK:            fn_name: mul
# CHECK:            operands:
# CHECK:              symbolic_cast:
# CHECK:                type_var: U
# CHECK:                operands:
# CHECK:                  scalar_arg: A
# CHECK:              symbolic_cast:
# CHECK:                type_var: U
# CHECK:                operands:
# CHECK:                  scalar_arg: B
@linalg_structured_op
def matmul(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


# CHECK: ---
# CHECK-LABEL: constants
# CHECK: assignments:
# CHECK:  -
# CHECK:    arg: O
# CHECK:      scalar_apply:
# CHECK:        fn_name: sub
# CHECK:        operands:
# CHECK:          scalar_apply:
# CHECK:            fn_name: add
# CHECK:            operands:
# CHECK:              symbolic_cast:
# CHECK:                type_var: T
# CHECK:                operands:
# CHECK:                  scalar_const: '3.1415926535897931 : f64'
# CHECK:              symbolic_cast:
# CHECK:                type_var: T
# CHECK:                operands:
# CHECK:                  scalar_const: '42 : i64'
# CHECK:          symbolic_cast:
# CHECK:            type_var: T
# CHECK:            operands:
# CHECK:              scalar_const: '1.{{[0]*}}e+03 : f64'
@linalg_structured_op
def constants(O=TensorDef(T, S.M, S.K, output=True)):
  pi = cast(T, const(3.1415926535897931))
  cst42 = cast(T, const(42))
  cst1000 = cast(T, const(1e+3))
  O[D.m, D.n] = pi + cst42 - cst1000


# CHECK: ---
# CHECK-LABEL: indices
# CHECK: assignments:
# CHECK:  -
# CHECK:    arg: O
# CHECK:      scalar_apply:
# CHECK:        fn_name: add
# CHECK:        operands:
# CHECK:          scalar_index: 1
# CHECK:          scalar_index: 0
@linalg_structured_op
def indices(O=TensorDef(T, S.M, S.K, output=True)):
  O[D.m, D.n] = index(D.n) + index(D.m)


# CHECK: ---
# CHECK-LABEL: fill
# CHECK: assignments:
# CHECK:  -
# CHECK:    arg: O
# CHECK:      scalar_arg: value
@linalg_structured_op
def fill(value=ScalarDef(T), O=TensorDef(T, S.M, S.K, output=True)):
  O[D.m, D.n] = value
