from ..lang import *

T1 = TV.T1
T2 = TV.T2

Batch = S.Batch


@linalg_structured_op
def copy(
    I=TensorDef(T1),
    O=TensorDef(U, output=True),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    """Copies the tensor elementwise.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    defines(Canonicalizer)
    O[None] = cast(U, I[None])


@linalg_structured_op
def elemwise_unary(
    I=TensorDef(T1),
    O=TensorDef(U, output=True),
    fun=UnaryFnAttrDef(default=UnaryFn.exp),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    """Applies the unary function fun elementwise.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    O[None] = fun(cast(U, I[None]))


@linalg_structured_op
def exp(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies exp(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.exp(I[None])


@linalg_structured_op
def log(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies log(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.log(I[None])


@linalg_structured_op
def abs(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies abs(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.abs(I[None])


@linalg_structured_op
def ceil(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies ceil(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.ceil(I[None])


@linalg_structured_op
def floor(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies floor(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.floor(I[None])


@linalg_structured_op(op_class_name="NegFOp")
def negf(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies negf(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.negf(I[None])


@linalg_structured_op(op_class_name="ReciprocalOp")
def reciprocal(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies reciprocal(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.reciprocal(I[None])


@linalg_structured_op
def round(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies round(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.round(I[None])


@linalg_structured_op
def sqrt(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies sqrt(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.sqrt(I[None])


@linalg_structured_op
def rsqrt(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies rsqrt(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.rsqrt(I[None])


@linalg_structured_op
def square(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies square(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.square(I[None])


@linalg_structured_op
def tanh(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies tanh(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.tanh(I[None])


@linalg_structured_op
def erf(
    I=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Applies erf(x) elementwise.

    No numeric casting is performed on the input operand.
    """
    O[None] = UnaryFn.erf(I[None])


@linalg_structured_op
def elemwise_binary(
    lhs=TensorDef(T1),
    rhs=TensorDef(T2),
    O=TensorDef(U, output=True),
    fun=BinaryFnAttrDef(default=BinaryFn.add),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    """Applies the binary function fun elementwise.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    O[None] = fun(cast(U, lhs[None]), cast(U, rhs[None]))


@linalg_structured_op
def add(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Adds two tensors elementwise.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.add` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.add(lhs[None], rhs[None])


@linalg_structured_op
def sub(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Subtracts two tensors elementwise.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.sub` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.sub(lhs[None], rhs[None])


@linalg_structured_op
def mul(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Multiplies two tensors elementwise.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.mul` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.mul(lhs[None], rhs[None])


@linalg_structured_op
def div(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Divides the first tensor by the second tensor, elementwise.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.div(lhs[None], rhs[None])


@linalg_structured_op
def div_unsigned(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Divides the first tensor by the second tensor, elementwise. For integer
    types, performs an unsigned division.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.div` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.div_unsigned(lhs[None], rhs[None])


@linalg_structured_op
def max(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Takes the max (signed) between two inputs, elementwise.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.max` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.max_signed(lhs[None], rhs[None])


@linalg_structured_op
def min(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Takes the min (signed) between two inputs, elementwise.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.min` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.min_signed(lhs[None], rhs[None])


@linalg_structured_op(op_class_name="PowFOp")
def powf(
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Takes the powf(lhs, rhs) between two inputs, elementwise. For powf(arg, 2) use `linalg.square`.

    Only applies to floating point values.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.powf` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = BinaryFn.powf(lhs[None], rhs[None])


@linalg_structured_op
def select(
    cond=TensorDef(U),
    lhs=TensorDef(T1),
    rhs=TensorDef(T1),
    O=TensorDef(T1, output=True),
):
    """Chooses one value based on a binary condition supplied as its first operand.

    The shapes and element types must be identical. The appropriate casts,
    broadcasts and reductions should be done previously to calling this op.

    This means reduction/broadcast/element cast semantics is explicit. Further
    passes can take that into account when lowering this code. For example,
    a `linalg.broadcast` + `linalg.select` sequence can be lowered to a
    `linalg.generic` with different affine maps for the two operands.
    """
    O[None] = TernaryFn.select(cond[None], lhs[None], rhs[None])


@linalg_structured_op
def quantized_matmul(
    A=TensorDef(T1, S.M, S.K),
    B=TensorDef(T2, S.K, S.N),
    AZp=ScalarDef(I32),
    BZp=ScalarDef(I32),
    C=TensorDef(U, S.M, S.N, output=True),
):
    """Performs a matrix multiplication of two 2D inputs.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. The quantized variant
    includes zero-point adjustments for the left and right operands of the
    matmul.
    """
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += (TypeFn.cast_signed(U, A[D.m, D.k]) - TypeFn.cast_signed(U, AZp)) * (
        TypeFn.cast_signed(U, B[D.k, D.n]) - TypeFn.cast_signed(U, BZp)
    )


@linalg_structured_op
def matmul_transpose_a(
    A=TensorDef(T1, S.K, S.N),
    B=TensorDef(T2, S.K, S.M),
    C=TensorDef(U, S.M, S.N, output=True),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    """Performs a matrix multiplication of two 2D inputs with lhs operand
    transposed.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.m, D.n] += cast(U, A[D.k, D.m]) * cast(U, B[D.k, D.n])


@linalg_structured_op
def matmul_transpose_b(
    A=TensorDef(T1, S.M, S.K),
    B=TensorDef(T2, S.N, S.K),
    C=TensorDef(U, S.M, S.N, output=True),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    """Performs a matrix multiplication of two 2D inputs with rhs operand
    transposed.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.n, D.k])


@linalg_structured_op
def mmt4d(
    lhs=TensorDef(TV.LhsType, S.M, S.K, S.M0, S.K0),
    rhs=TensorDef(TV.RhsType, S.N, S.K, S.N0, S.K0),
    accum=TensorDef(TV.AccumType, S.M, S.N, S.M0, S.N0, output=True),
):
    """Performs a matrix-matrix-transpose multiplication of two 4D inputs.

    Differences from linalg.matmul:
    * The right hand side is transposed, whence the 't' in 'mmt'.
    * The input and output tensors have a 4D shape instead of a 2D shape. They
      are interpreted as 2D matrices with one level of 2D tile subdivision,
      whence the 2+2=4 dimensions. The inner tile dimensions are identified with
      '0' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
      as: MxK tiles, each of shape M0xK0.
    """
    domain(D.m, D.n, D.k, D.m0, D.n0, D.k0)
    implements(ContractionOpInterface)
    accum[D.m, D.n, D.m0, D.n0] += TypeFn.cast_signed(
        TV.AccumType, lhs[D.m, D.k, D.m0, D.k0]
    ) * TypeFn.cast_signed(TV.AccumType, rhs[D.n, D.k, D.n0, D.k0])


@linalg_structured_op
def batch_mmt4d(
    lhs=TensorDef(TV.LhsType, Batch, S.M, S.K, S.M0, S.K0),
    rhs=TensorDef(TV.RhsType, Batch, S.N, S.K, S.N0, S.K0),
    accum=TensorDef(TV.AccumType, Batch, S.M, S.N, S.M0, S.N0, output=True),
):
    """Performs a batched matrix-matrix-transpose multiplication of two
    batched-4D (5D) inputs.

    Besides the outermost batch dimension has the same semantic as
    linalg.batch_matmul, the differences from linalg.batch_matmul in the
    non-batch dimensions are the same as linalg.mmt4d vs. linalg.matmul. See the
    description of lingalg.mmt4d.
    """
    domain(D.b, D.m, D.n, D.k, D.m0, D.n0, D.k0)
    implements(ContractionOpInterface)
    accum[D.b, D.m, D.n, D.m0, D.n0] += TypeFn.cast_signed(
        TV.AccumType, lhs[D.b, D.m, D.k, D.m0, D.k0]
    ) * TypeFn.cast_signed(TV.AccumType, rhs[D.b, D.n, D.k, D.n0, D.k0])


@linalg_structured_op
def batch_matmul(
    A=TensorDef(T1, Batch, S.M, S.K),
    B=TensorDef(T2, Batch, S.K, S.N),
    C=TensorDef(U, Batch, S.M, S.N, output=True),
):
    """Performs a batched matrix multiplication of two 3D inputs.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.b, D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.b, D.m, D.n] += TypeFn.cast_signed(U, A[D.b, D.m, D.k]) * TypeFn.cast_signed(
        U, B[D.b, D.k, D.n]
    )


@linalg_structured_op
def batch_matmul_transpose_a(
    A=TensorDef(T1, Batch, S.K, S.M),
    B=TensorDef(T2, Batch, S.K, S.N),
    C=TensorDef(U, Batch, S.M, S.N, output=True),
):
    """Performs a batched matrix multiplication of two 3D inputs where lhs operand
    has its non-batch dimensions transposed.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.b, D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.b, D.m, D.n] += TypeFn.cast_signed(U, A[D.b, D.k, D.m]) * TypeFn.cast_signed(
        U, B[D.b, D.k, D.n]
    )


@linalg_structured_op
def batch_matmul_transpose_b(
    A=TensorDef(T1, Batch, S.M, S.K),
    B=TensorDef(T2, Batch, S.N, S.K),
    C=TensorDef(U, Batch, S.M, S.N, output=True),
):
    """Performs a batched matrix multiplication of two 3D inputs where rhs operand
    has its non-batch dimensions transposed.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.b, D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.b, D.m, D.n] += TypeFn.cast_signed(U, A[D.b, D.m, D.k]) * TypeFn.cast_signed(
        U, B[D.b, D.n, D.k]
    )


@linalg_structured_op
def quantized_batch_matmul(
    A=TensorDef(T1, Batch, S.M, S.K),
    B=TensorDef(T2, Batch, S.K, S.N),
    AZp=ScalarDef(I32),
    BZp=ScalarDef(I32),
    C=TensorDef(U, Batch, S.M, S.N, output=True),
):
    """Performs a batched matrix multiplication of two 3D inputs.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. The quantized variant
    includes zero-point adjustments for the left and right operands of the
    matmul.
    """
    domain(D.b, D.m, D.n, D.k)
    C[D.b, D.m, D.n] += (
        TypeFn.cast_signed(U, A[D.b, D.m, D.k]) - TypeFn.cast_signed(U, AZp)
    ) * (TypeFn.cast_signed(U, B[D.b, D.k, D.n]) - TypeFn.cast_signed(U, BZp))


@linalg_structured_op
def batch_reduce_matmul(
    A=TensorDef(T1, Batch, S.M, S.K),
    B=TensorDef(T2, Batch, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
):
    """Performs a batch-reduce matrix multiplication of two 3D inputs.
    The partial multiplication results are reduced into a 2D output.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.b, D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.m, D.n] += TypeFn.cast_signed(U, A[D.b, D.m, D.k]) * TypeFn.cast_signed(
        U, B[D.b, D.k, D.n]
    )


@linalg_structured_op
def matvec(
    A=TensorDef(T1, S.M, S.N), y=TensorDef(T2, S.N), x=TensorDef(U, S.M, output=True)
):
    """Performs a matrix-vector multiplication.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.m, D.n)
    implements(ContractionOpInterface)
    x[D.m] += TypeFn.cast_signed(U, A[D.m, D.n]) * TypeFn.cast_signed(U, y[D.n])


@linalg_structured_op
def vecmat(
    y=TensorDef(T1, S.M), A=TensorDef(T2, S.M, S.N), x=TensorDef(U, S.N, output=True)
):
    """Performs a vector-matrix multiplication.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.n, D.m)
    implements(ContractionOpInterface)
    x[D.n] += TypeFn.cast_signed(U, y[D.m]) * TypeFn.cast_signed(U, A[D.m, D.n])


@linalg_structured_op
def batch_matvec(
    A=TensorDef(T1, Batch, S.M, S.K),
    B=TensorDef(T2, Batch, S.K),
    C=TensorDef(U, Batch, S.M, output=True),
):
    """Performs a batched matrix-vector multiplication.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.b, D.m, D.k)
    implements(ContractionOpInterface)
    C[D.b, D.m] += TypeFn.cast_signed(U, A[D.b, D.m, D.k]) * TypeFn.cast_signed(
        U, B[D.b, D.k]
    )


@linalg_structured_op
def batch_vecmat(
    A=TensorDef(T1, Batch, S.K),
    B=TensorDef(T2, Batch, S.K, S.N),
    C=TensorDef(U, Batch, S.N, output=True),
):
    """Performs a batched matrix-vector multiplication.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    domain(D.b, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.b, D.n] += TypeFn.cast_signed(U, A[D.b, D.k]) * TypeFn.cast_signed(
        U, B[D.b, D.k, D.n]
    )


@linalg_structured_op
def dot(A=TensorDef(T1, S.M), B=TensorDef(T2, S.M), C=TensorDef(U, output=True)):
    """Performs a dot product of two vectors to a scalar result.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ContractionOpInterface)
    C[None] += TypeFn.cast_signed(U, A[D.m]) * TypeFn.cast_signed(U, B[D.m])


@linalg_structured_op
def conv_1d(
    I=TensorDef(T1, S.OW + S.KW),
    K=TensorDef(T2, S.KW),
    O=TensorDef(U, S.OW, output=True),
):
    """Performs 1-D convolution with no channels.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.ow, D.kw)
    O[D.ow] += TypeFn.cast_signed(U, I[D.ow + D.kw]) * TypeFn.cast_signed(U, K[D.kw])


@linalg_structured_op
def conv_2d(
    I=TensorDef(T1, S.OH + S.KH, S.OW + S.KW),
    K=TensorDef(T2, S.KH, S.KW),
    O=TensorDef(U, S.OH, S.OW, output=True),
):
    """Performs 2-D convolution with no channels.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.oh, D.ow, D.kh, D.kw)
    O[D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.oh + D.kh, D.ow + D.kw]
    ) * TypeFn.cast_signed(U, K[D.kh, D.kw])


@linalg_structured_op
def conv_3d(
    I=TensorDef(T1, S.OD + S.KD, S.OH + S.KH, S.OW + S.KW),
    K=TensorDef(T2, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.OD, S.OH, S.OW, output=True),
):
    """Performs 3-D convolution with no channels.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.od, D.oh, D.ow, D.kd, D.kh, D.kw)
    O[D.od, D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.od + D.kd, D.oh + D.kh, D.ow + D.kw]
    ) * TypeFn.cast_signed(U, K[D.kd, D.kh, D.kw])


@linalg_structured_op
def conv_1d_nwc_wcf(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs 1-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.f, D.kw, D.c)
    O[D.n, D.ow, D.f] += TypeFn.cast_signed(
        U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c]
    ) * TypeFn.cast_signed(U, K[D.kw, D.c, D.f])


@linalg_structured_op
def conv_1d_ncw_fcw(
    I=TensorDef(T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.F, S.C, S.KW),
    O=TensorDef(U, S.N, S.F, S.OW, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs 1-D convolution.

    Layout:
      * Input: NCW.
      * Kernel: FCW.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.f, D.ow, D.c, D.kw)
    O[D.n, D.f, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW]
    ) * TypeFn.cast_signed(U, K[D.f, D.c, D.kw])


@linalg_structured_op
def conv_2d_nhwc_hwcf(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D convolution.

    Layout:
      * Input: NHWC.
      * Kernel: HWCF.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
    O[D.n, D.oh, D.ow, D.f] += TypeFn.cast_signed(
        U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
    ) * TypeFn.cast_signed(U, K[D.kh, D.kw, D.c, D.f])


@linalg_structured_op
def conv_2d_nhwc_fhwc(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.F, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D convolution.

    Layout:
      * Input: NHWC.
      * Kernel: FHWC.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
    O[D.n, D.oh, D.ow, D.f] += TypeFn.cast_signed(
        U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
    ) * TypeFn.cast_signed(U, K[D.f, D.kh, D.kw, D.c])


@linalg_structured_op
def conv_2d_nhwc_hwcf_q(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, S.C, S.F),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D convolution with zero point offsets.

    Layout:
      * Input: NHWC.
      * Kernel: HWCF.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. This includes the zero
    point offsets common to quantized operations.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
    O[D.n, D.oh, D.ow, D.f] += (
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (TypeFn.cast_signed(U, K[D.kh, D.kw, D.c, D.f]) - TypeFn.cast_signed(U, KZp))


@linalg_structured_op
def conv_2d_nhwc_fhwc_q(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.F, S.KH, S.KW, S.C),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.OH, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D convolution with zero point offsets.

    Layout:
      * Input: NHWC.
      * Kernel: FHWC.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. This includes the zero
    point offsets common to quantized operations.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.f, D.kh, D.kw, D.c)
    O[D.n, D.oh, D.ow, D.f] += (
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (TypeFn.cast_signed(U, K[D.f, D.kh, D.kw, D.c]) - TypeFn.cast_signed(U, KZp))


@linalg_structured_op
def conv_2d_nchw_fchw_q(
    I=TensorDef(T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.F, S.C, S.KH, S.KW),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D convolution with zero point offsets.

    Layout:
      * Input: NCHW.
      * Kernel: FCHW.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. This includes the zero
    point offsets common to quantized operations.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.f, D.oh, D.ow] += (
        TypeFn.cast_signed(
            U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (TypeFn.cast_signed(U, K[D.f, D.c, D.kh, D.kw]) - TypeFn.cast_signed(U, KZp))

@linalg_structured_op
def conv_2d_nchw_fchw(
    I=TensorDef(T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D convolution.

    Layout:
      * Input: NCHW.
      * Kernel: FCHW.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.f, D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
    ) * TypeFn.cast_signed(U, K[D.f, D.c, D.kh, D.kw])


@linalg_structured_op
def conv_2d_ngchw_fgchw(
    I=TensorDef(
        T1, S.N, S.G, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW
    ),
    K=TensorDef(T2, S.FG, S.G, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.G, S.FG, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D grouped convolution.

    Layout:
      * Input: NGCHW.
      * Kernel: FGCHW.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.g, D.fg, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.g, D.fg, D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.g, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
    ) * TypeFn.cast_signed(U, K[D.fg, D.g, D.c, D.kh, D.kw])


@linalg_structured_op
def conv_2d_ngchw_gfchw(
    I=TensorDef(
        T1, S.N, S.G, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW
    ),
    K=TensorDef(T2, S.G, S.FG, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.G, S.FG, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D grouped convolution.

    Layout:
      * Input: NGCHW.
      * Kernel: GFCHW.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.g, D.fg, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.g, D.fg, D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.g, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
    ) * TypeFn.cast_signed(U, K[D.g, D.fg, D.c, D.kh, D.kw])


@linalg_structured_op
def conv_2d_nhwgc_gfhwc(
    I=TensorDef(
        T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.G, S.C
    ),
    K=TensorDef(T2, S.G, S.FG, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.G, S.FG, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D grouped convolution.

    Layout:
      * Input: NHWGC.
      * Kernel: GFHWC.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.g, D.fg, D.kh, D.kw, D.c)
    O[D.n, D.oh, D.ow, D.g, D.fg] += TypeFn.cast_signed(
        U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.g, D.c]
    ) * TypeFn.cast_signed(U, K[D.g, D.fg, D.kh, D.kw, D.c])


@linalg_structured_op
def conv_2d_nhwgc_gfhwc_q(
    I=TensorDef(
        T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.G, S.C
    ),
    K=TensorDef(T2, S.G, S.FG, S.KH, S.KW, S.C),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.OH, S.OW, S.G, S.FG, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D grouped convolution with zero point offsets.

    Layout:
      * Input: NHWGC.
      * Kernel: GFHWC.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. This includes the zero
    point offsets common to quantized operations.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.g, D.fg, D.kh, D.kw, D.c)
    O[D.n, D.oh, D.ow, D.g, D.fg] += (
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.g, D.c]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (
        TypeFn.cast_signed(U, K[D.g, D.fg, D.kh, D.kw, D.c])
        - TypeFn.cast_signed(U, KZp)
    )


@linalg_structured_op
def conv_2d_ngchw_gfchw_q(
    I=TensorDef(
        T1, S.N, S.G, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW
    ),
    K=TensorDef(T2, S.G, S.FG, S.C, S.KH, S.KW),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.G, S.FG, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs 2-D grouped convolution with zero-point offsets.

    Layout:
      * Input: NGCHW.
      * Kernel: GFCHW.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. This includes the zero
    point offsets common to quantized operations.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.g, D.fg, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.g, D.fg, D.oh, D.ow] += (
        TypeFn.cast_signed(
            U, I[D.n, D.g, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (
        TypeFn.cast_signed(U, K[D.g, D.fg, D.c, D.kh, D.kw])
        - TypeFn.cast_signed(U, KZp)
    )


@linalg_structured_op
def conv_3d_ndhwc_dhwcf(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.C,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, S.C, S.F),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
    O[D.n, D.od, D.oh, D.ow, D.f] += TypeFn.cast_signed(
        U,
        I[
            D.n,
            D.od * S.SD + D.kd * S.DD,
            D.oh * S.SH + D.kh * S.DH,
            D.ow * S.SW + D.kw * S.DW,
            D.c,
        ],
    ) * TypeFn.cast_signed(U, K[D.kd, D.kh, D.kw, D.c, D.f])


@linalg_structured_op
def conv_3d_ndhwc_dhwcf_q(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.C,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, S.C, S.F),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.F, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3-D convolution with zero point offsets.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. This includes the zero
    point offsets common to quantized operations.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
    O[D.n, D.od, D.oh, D.ow, D.f] += (
        TypeFn.cast_signed(
            U,
            I[
                D.n,
                D.od * S.SD + D.kd * S.DD,
                D.oh * S.SH + D.kh * S.DH,
                D.ow * S.SW + D.kw * S.DW,
                D.c,
            ],
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (
        TypeFn.cast_signed(U, K[D.kd, D.kh, D.kw, D.c, D.f])
        - TypeFn.cast_signed(U, KZp)
    )


@linalg_structured_op
def conv_3d_ncdhw_fcdhw(
    I=TensorDef(
        T1,
        S.N,
        S.C,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
    ),
    K=TensorDef(T2, S.F, S.C, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OD, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.f, D.kd, D.kh, D.kw, D.c)
    O[D.n, D.f, D.od, D.oh, D.ow] += TypeFn.cast_signed(
        U,
        I[
            D.n,
            D.c,
            D.od * S.SD + D.kd * S.DD,
            D.oh * S.SH + D.kh * S.DH,
            D.ow * S.SW + D.kw * S.DW,
        ],
    ) * TypeFn.cast_signed(U, K[D.f, D.c, D.kd, D.kh, D.kw])


@linalg_structured_op
def depthwise_conv_1d_nwc_wc(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.IC),
    K=TensorDef(T2, S.KW, S.IC),
    O=TensorDef(U, S.N, S.OW, S.IC, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs depth-wise 1-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. Multiplier is set to 1
    which is a special case for most depthwise convolutions.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.ic, D.kw)
    O[D.n, D.ow, D.ic] += TypeFn.cast_signed(
        U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.ic]
    ) * TypeFn.cast_signed(U, K[D.kw, D.ic])


@linalg_structured_op
def depthwise_conv_1d_ncw_cw(
    I=TensorDef(T1, S.N, S.IC, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.IC, S.KW),
    O=TensorDef(U, S.N, S.IC, S.OW, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs depth-wise 1-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. Multiplier is set to 1
    which is a special case for most depthwise convolutions.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.ic, D.kw)
    O[D.n, D.ic, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.ic, D.ow * S.SW + D.kw * S.DW]
    ) * TypeFn.cast_signed(U, K[D.ic, D.kw])


@linalg_structured_op
def depthwise_conv_1d_nwc_wcm(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.IC),
    K=TensorDef(T2, S.KW, S.IC, S.CM),
    O=TensorDef(U, S.N, S.OW, S.IC, S.CM, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs depth-wise 1-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.ic, D.cm, D.kw)
    O[D.n, D.ow, D.ic, D.cm] += TypeFn.cast_signed(
        U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.ic]
    ) * TypeFn.cast_signed(U, K[D.kw, D.ic, D.cm])


@linalg_structured_op
def depthwise_conv_2d_nhwc_hwc(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.IC),
    K=TensorDef(T2, S.KH, S.KW, S.IC),
    O=TensorDef(U, S.N, S.OH, S.OW, S.IC, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs depth-wise 2-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. Multiplier is set to 1
    which is a special case for most depthwise convolutions.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.ic, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.ic] += TypeFn.cast_signed(
        U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.ic]
    ) * TypeFn.cast_signed(U, K[D.kh, D.kw, D.ic])


@linalg_structured_op
def depthwise_conv_2d_nchw_chw(
    I=TensorDef(T1, S.N, S.IC, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.IC, S.KH, S.KW),
    O=TensorDef(U, S.N, S.IC, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs depth-wise 2-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. Multiplier is set to 1
    which is a special case for most depthwise convolutions.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.ic, D.kh, D.kw)
    O[D.n, D.ic, D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.ic, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
    ) * TypeFn.cast_signed(U, K[D.ic, D.kh, D.kw])


@linalg_structured_op
def depthwise_conv_2d_nhwc_hwc_q(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.IC),
    K=TensorDef(T2, S.KH, S.KW, S.IC),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.OH, S.OW, S.IC, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs depth-wise 2-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.ic, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.ic] += (
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.ic]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (TypeFn.cast_signed(U, K[D.kh, D.kw, D.ic]) - TypeFn.cast_signed(U, KZp))


@linalg_structured_op
def depthwise_conv_2d_nhwc_hwcm(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.IC),
    K=TensorDef(T2, S.KH, S.KW, S.IC, S.CM),
    O=TensorDef(U, S.N, S.OH, S.OW, S.IC, S.CM, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs depth-wise 2-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.ic, D.cm, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.ic, D.cm] += TypeFn.cast_signed(
        U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.ic]
    ) * TypeFn.cast_signed(U, K[D.kh, D.kw, D.ic, D.cm])


@linalg_structured_op
def depthwise_conv_2d_nhwc_hwcm_q(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.IC),
    K=TensorDef(T2, S.KH, S.KW, S.IC, S.CM),
    IZp=ScalarDef(I32),
    KZp=ScalarDef(I32),
    O=TensorDef(U, S.N, S.OH, S.OW, S.IC, S.CM, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs depth-wise 2-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.ic, D.cm, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.ic, D.cm] += (
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.ic]
        )
        - TypeFn.cast_signed(U, IZp)
    ) * (TypeFn.cast_signed(U, K[D.kh, D.kw, D.ic, D.cm]) - TypeFn.cast_signed(U, KZp))


@linalg_structured_op
def depthwise_conv_3d_ndhwc_dhwc(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.IC,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, S.IC),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs depth-wise 3-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. Multiplier is set to 1
    which is a special case for most depthwise convolutions.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.kd, D.kh, D.kw, D.ic)
    O[D.n, D.od, D.oh, D.ow, D.ic] += TypeFn.cast_signed(
        U,
        I[
            D.n,
            D.od * S.SD + D.kd * S.DD,
            D.oh * S.SH + D.kh * S.DH,
            D.ow * S.SW + D.kw * S.DW,
            D.ic,
        ],
    ) * TypeFn.cast_signed(U, K[D.kd, D.kh, D.kw, D.ic])


@linalg_structured_op
def depthwise_conv_3d_ncdhw_cdhw(
    I=TensorDef(
        T1,
        S.N,
        S.IC,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
    ),
    K=TensorDef(T2, S.IC, S.KD, S.KH, S.KW),
    O=TensorDef(U, S.N, S.IC, S.OD, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs depth-wise 3-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output. Multiplier is set to 1
    which is a special case for most depthwise convolutions.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.kd, D.kh, D.kw, D.ic)
    O[D.n, D.ic, D.od, D.oh, D.ow] += TypeFn.cast_signed(
        U,
        I[
            D.n,
            D.ic,
            D.od * S.SD + D.kd * S.DD,
            D.oh * S.SH + D.kh * S.DH,
            D.ow * S.SW + D.kw * S.DW,
        ],
    ) * TypeFn.cast_signed(U, K[D.ic, D.kd, D.kh, D.kw])


@linalg_structured_op
def depthwise_conv_3d_ndhwc_dhwcm(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.IC,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, S.IC, S.CM),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.CM, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs depth-wise 3-D convolution.

    Numeric casting is performed on the operands to the inner multiply, promoting
    them to the same data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.cm, D.kd, D.kh, D.kw, D.ic)
    O[D.n, D.od, D.oh, D.ow, D.ic, D.cm] += TypeFn.cast_signed(
        U,
        I[
            D.n,
            D.od * S.SD + D.kd * S.DD,
            D.oh * S.SH + D.kh * S.DH,
            D.ow * S.SW + D.kw * S.DW,
            D.ic,
        ],
    ) * TypeFn.cast_signed(U, K[D.kd, D.kh, D.kw, D.ic, D.cm])


@linalg_structured_op
def pooling_nhwc_sum(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs sum pooling.

    Layout:
      * Input: NHWC.
      * Kernel: HW.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.c] += TypeFn.cast_signed(
        U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
    )


@linalg_structured_op
def pooling_nchw_sum(
    I=TensorDef(T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.C, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs sum pooling.

    Layout:
      * Input: NCHW.
      * Kernel: HW.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.c, D.oh, D.ow, D.kh, D.kw)
    O[D.n, D.c, D.oh, D.ow] += TypeFn.cast_signed(
        U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW]
    )


@linalg_structured_op
def pooling_nhwc_max(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.c] = ReduceFn.max_signed[D.kh, D.kw](
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
        )
    )


@linalg_structured_op
def pooling_nhwc_max_unsigned(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs unsigned max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.c] = ReduceFn.max_unsigned[D.kh, D.kw](
        TypeFn.cast_unsigned(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
        )
    )


@linalg_structured_op
def pooling_nchw_max(
    I=TensorDef(T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.C, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.c, D.oh, D.ow, D.kh, D.kw)
    O[D.n, D.c, D.oh, D.ow] = ReduceFn.max_signed[D.kh, D.kw](
        TypeFn.cast_signed(
            U,
            I[
                D.n,
                D.c,
                D.oh * S.SH + D.kh * S.DH,
                D.ow * S.SW + D.kw * S.DW,
            ],
        )
    )


@linalg_structured_op
def pooling_nhwc_min(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs min pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.c] = ReduceFn.min_signed[D.kh, D.kw](
        TypeFn.cast_signed(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
        )
    )


@linalg_structured_op
def pooling_nhwc_min_unsigned(
    I=TensorDef(T1, S.N, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 1]),
):
    """Performs unsigned min pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.oh, D.ow, D.c, D.kh, D.kw)
    O[D.n, D.oh, D.ow, D.c] = ReduceFn.min_unsigned[D.kh, D.kw](
        TypeFn.cast_unsigned(
            U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]
        )
    )


@linalg_structured_op
def pooling_nwc_sum(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs sum pooling.

    Layout:
      * Input: NWC.
      * Kernel: W.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.c, D.kw)
    O[D.n, D.ow, D.c] += TypeFn.cast_signed(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c])


@linalg_structured_op
def pooling_ncw_sum(
    I=TensorDef(T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.C, S.OW, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs sum pooling.

    Layout:
      * Input: NCW.
      * Kernel: W.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.c, D.ow, D.kw)
    O[D.n, D.c, D.ow] += TypeFn.cast_signed(U, I[D.n, D.c, D.ow * S.SW + D.kw * S.DW])


@linalg_structured_op
def pooling_nwc_max(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.c, D.kw)
    O[D.n, D.ow, D.c] = ReduceFn.max_signed[[D.kw]](
        TypeFn.cast_signed(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c])
    )


@linalg_structured_op
def pooling_nwc_max_unsigned(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs unsigned max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.c, D.kw)
    O[D.n, D.ow, D.c] = ReduceFn.max_unsigned[[D.kw]](
        TypeFn.cast_unsigned(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c])
    )


@linalg_structured_op
def pooling_ncw_max(
    I=TensorDef(T1, S.N, S.C, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.C, S.OW, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.c, D.ow, D.kw)
    O[D.n, D.c, D.ow] = ReduceFn.max_signed[[D.kw]](
        TypeFn.cast_signed(
            U,
            I[
                D.n,
                D.c,
                D.ow * S.SW + D.kw * S.DW,
            ],
        )
    )


@linalg_structured_op
def pooling_nwc_min(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs min pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.c, D.kw)
    O[D.n, D.ow, D.c] = ReduceFn.min_signed[[D.kw]](
        TypeFn.cast_signed(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c])
    )


@linalg_structured_op
def pooling_nwc_min_unsigned(
    I=TensorDef(T1, S.N, S.OW * S.SW + S.KW * S.DW, S.C),
    K=TensorDef(T2, S.KW, index_dims=[D.kw]),
    O=TensorDef(U, S.N, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SW, default=[1]),
    dilations=IndexAttrDef(S.DW, default=[1]),
):
    """Performs unsigned min pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.ow, D.c, D.kw)
    O[D.n, D.ow, D.c] = ReduceFn.min_unsigned[[D.kw]](
        TypeFn.cast_unsigned(U, I[D.n, D.ow * S.SW + D.kw * S.DW, D.c])
    )


@linalg_structured_op
def pooling_ndhwc_sum(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.C,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, index_dims=[D.kd, D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3D sum pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
    O[D.n, D.od, D.oh, D.ow, D.c] += TypeFn.cast_signed(
        U,
        I[
            D.n,
            D.od * S.SD + D.kd * S.DD,
            D.oh * S.SH + D.kh * S.DH,
            D.ow * S.SW + D.kw * S.DW,
            D.c,
        ],
    )


@linalg_structured_op
def pooling_ndhwc_max(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.C,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, index_dims=[D.kd, D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3D max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
    O[D.n, D.od, D.oh, D.ow, D.c] = ReduceFn.max_signed[D.kd, D.kh, D.kw](
        TypeFn.cast_signed(
            U,
            I[
                D.n,
                D.od * S.SD + D.kd * S.DD,
                D.oh * S.SH + D.kh * S.DH,
                D.ow * S.SW + D.kw * S.DW,
                D.c,
            ],
        )
    )


@linalg_structured_op
def pooling_ndhwc_min(
    I=TensorDef(
        T1,
        S.N,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
        S.C,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, index_dims=[D.kd, D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OD, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3D min pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.od, D.oh, D.ow, D.c, D.kd, D.kh, D.kw)
    O[D.n, D.od, D.oh, D.ow, D.c] = ReduceFn.min_signed[D.kd, D.kh, D.kw](
        TypeFn.cast_signed(
            U,
            I[
                D.n,
                D.od * S.SD + D.kd * S.DD,
                D.oh * S.SH + D.kh * S.DH,
                D.ow * S.SW + D.kw * S.DW,
                D.c,
            ],
        )
    )


@linalg_structured_op
def fill(value=ScalarDef(T1), O=TensorDef(U, output=True)):
    """Fills the output tensor with the given value.

    Works for arbitrary ranked output tensors since the operation performs scalar
    accesses only and is thus rank polymorphic. Numeric casting is performed on
    the value operand, promoting it to the same data type as the output.
    """
    implements(FillOpInterface)
    defines(Canonicalizer)
    O[None] = TypeFn.cast_signed(U, value)


@linalg_structured_op
def fill_rng_2d(
    min=ScalarDef(F64),
    max=ScalarDef(F64),
    seed=ScalarDef(I32),
    O=TensorDef(T, S.M, S.N, output=True),
):
    """Fills the output tensor with pseudo random numbers.

    The operation generations pseudo random numbers using a linear congruential
    generator. It provides no guarantees regarding the distribution of the
    generated random numbers. Instead of generating the random numbers
    sequentially, it instantiates one random number generator per data element
    and runs them in parallel. The seed operand and the indices of the data
    element seed the random number generation. The min and max operands limit
    the range of the generated random numbers.
    """
    domain(D.m, D.n)
    multiplier = TypeFn.cast_signed(I32, const(1103515245))
    increment = TypeFn.cast_signed(I32, const(12345))
    rand1 = (TypeFn.cast_signed(I32, index(D.m)) + seed) * multiplier + increment
    rand2 = (TypeFn.cast_signed(I32, index(D.n)) + rand1) * multiplier + increment
    inv_range = TypeFn.cast_signed(F64, const(2.3283064e-10))
    offset = TypeFn.cast_signed(F64, const(2147483647))
    scaling = (max - min) * inv_range
    O[D.m, D.n] = TypeFn.cast_signed(
        T, (offset + TypeFn.cast_signed(F64, rand2)) * scaling + min
    )
