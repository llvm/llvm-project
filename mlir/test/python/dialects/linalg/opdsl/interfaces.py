# RUN: %PYTHON -m mlir.dialects.linalg.opdsl.dump_oplib --file %s | FileCheck %s

from mlir.dialects.linalg.opdsl.lang import *


# CHECK: ---
# CHECK-LABEL: matmul
# CHECK:      implements:
# CHECK-NEXT: - LinalgContractionOpInterface
@linalg_structured_op
def matmul(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True)):
  implements(ContractionOpInterface)
  C[D.m, D.n] += TypeFn.cast_signed(U, A[D.m, D.k]) * TypeFn.cast_signed(
      U, B[D.k, D.n])
