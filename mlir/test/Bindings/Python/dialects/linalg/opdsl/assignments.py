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
def matmul(A=TensorDef(T, S.M, S.K),
           B=TensorDef(T, S.K, S.N),
           C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])
