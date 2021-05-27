from ..lang import *

T1 = TV.T1
T2 = TV.T2

Batch = S.Batch


@linalg_structured_op
def matmul(
    A=TensorDef(T1, S.M, S.K),
    B=TensorDef(T2, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True)):
  """Performs a matrix multiplication of two 2D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


@linalg_structured_op
def batch_matmul(
    A=TensorDef(T1, Batch, S.M, S.K),
    B=TensorDef(T2, Batch, S.K, S.N),
    C=TensorDef(U, Batch, S.M, S.N, output=True)):
  """Performs a batched matrix multiplication of two 3D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[D.b, D.m, D.n] += cast(U, A[D.b, D.m, D.k]) * cast(U, B[D.b, D.k, D.n])


@linalg_structured_op
def matvec(
    A=TensorDef(T1, S.M, S.N),
    y=TensorDef(T2, S.N),
    x=TensorDef(U, S.M, output=True)):
  """Performs a matrix-vector multiplication.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  x[D.m] += cast(U, A[D.m, D.n]) * cast(U, y[D.n])


@linalg_structured_op
def vecmat(
    y=TensorDef(T1, S.M),
    A=TensorDef(T2, S.M, S.N),
    x=TensorDef(U, S.N, output=True)):
  """Performs a vector-matrix multiplication.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  x[D.n] += cast(U, y[D.m]) * cast(U, A[D.m, D.n])


@linalg_structured_op
def dot(
    A=TensorDef(T1, S.M), B=TensorDef(T2, S.M), C=TensorDef(U, output=True)):
  """Performs a dot product of two vectors to a scalar result.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ContractionOpInterface)
  C[None] += cast(U, A[D.m]) * cast(U, B[D.m])


@linalg_structured_op
def fill_rng_2d(O=TensorDef(T, S.M, S.N, output=True)):
  """Fills the output tensor with pseudo random numbers.

  The operation generations pseudo random numbers using a linear congruential
  generator. It provides no guarantees regarding the distribution of the
  generated random numbers. Instead of generating the random numbers
  sequentially, it instantiates one random number generator per data element
  and runs them in parallel. The seed operand and the indices of the data
  element seed the random number generation. The min and max operands limit
  the range of the generated random numbers.

  Note: The captures are hard-coded till there is capture support on the C++
  side.
  """
  min = cast(F64, const(-1000))
  max = cast(F64, const(+1000))
  seed = cast(I32, const(42))
  multiplier = cast(I32, const(1103515245))
  increment = cast(I32, const(12345))
  rand1 = (cast(I32, index(D.m)) + seed) * multiplier + increment
  rand2 = (cast(I32, index(D.n)) + rand1) * multiplier + increment
  inv_range = cast(F64, const(2.3283064e-10))
  offset = cast(F64, const(2147483647))
  scaling = (max - min) * inv_range
  O[D.m, D.n] = cast(T, (offset + cast(F64, rand2)) * scaling + min)
