# RUN: %PYTHON -m mlir.tools.linalg_opdsl.dump_oplib --file %s | FileCheck %s

from mlir.tools.linalg_opdsl.lang import *

# CHECK: ---
# CHECK-LABEL: matmul
# CHECK:      implements:
# CHECK-NEXT: - LinalgContractionOpInterface
@linalg_structured_op
def matmul(A=TensorDef(T, S.M, S.K),
           B=TensorDef(T, S.K, S.N),
           C=TensorDef(U, S.M, S.N, output=True)):
  implements(ContractionOpInterface)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])
