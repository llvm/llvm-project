from ..lang import *

T1 = TV.T1
T2 = TV.T2

Batch = S.Batch


@linalg_structured_op
def matmul(A=TensorDef(T1, S.M, S.K),
           B=TensorDef(T2, S.K, S.N),
           C=TensorDef(U, S.M, S.N, output=True)):
  """Performs a matrix multiplacation of two 2D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


@linalg_structured_op
def batch_matmul(A=TensorDef(T1, Batch, S.M, S.K),
                 B=TensorDef(T2, Batch, S.K, S.N),
                 C=TensorDef(U, Batch, S.M, S.N, output=True)):
  """Performs a batched matrix multiplacation of two 3D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[D.b, D.m, D.n] += cast(U, A[D.b, D.m, D.k]) * cast(U, B[D.b, D.k, D.n])


@linalg_structured_op
def matvec(A=TensorDef(T1, S.M, S.N),
           y=TensorDef(T2, S.N),
           x=TensorDef(U, S.M, output=True)):
  """Performs a matrix-vector multiplication.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  x[D.m] += cast(U, A[D.m, D.n]) * cast(U, y[D.n])


@linalg_structured_op
def vecmat(y=TensorDef(T1, S.M),
           A=TensorDef(T2, S.M, S.N),
           x=TensorDef(U, S.N, output=True)):
  """Performs a vector-matrix multiplacation.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  x[D.n] += cast(U, y[D.m]) * cast(U, A[D.m, D.n])


@linalg_structured_op
def dot(A=TensorDef(T1, S.M), B=TensorDef(T2, S.M), C=TensorDef(U,
                                                                output=True)):
  """Performs a dot product of two vectors to a scalar result.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[None] += cast(U, A[D.m]) * cast(U, B[D.m])
