# RUN: %PYTHON -m mlir.dialects.linalg.opdsl.dump_oplib --file %s | FileCheck %s

from mlir.dialects.linalg.opdsl.lang import *


# CHECK: ---
# CHECK-LABEL: matmul
# CHECK: args:
# CHECK:     name: A
# CHECK:     kind: input_tensor
# CHECK:     type_var: T
# CHECK:     shape_map: affine_map<()[s0, s1, s2] -> (s0, s1)>
# CHECK:     name: B
# CHECK:     kind: input_tensor
# CHECK:     type_var: T
# CHECK:     shape_map: affine_map<()[s0, s1, s2] -> (s1, s2)>
# CHECK:     name: C
# CHECK:     kind: output_tensor
# CHECK:     type_var: U
# CHECK:     shape_map: affine_map<()[s0, s1, s2] -> (s0, s2)>
# CHECK:     name: bfn
# CHECK:     kind: binary_fn_attr
# CHECK:     default_fn: mul
# CHECK:     name: ufn
# CHECK:     kind: unary_fn_attr
# CHECK:     default_fn: exp
# CHECK:     name: cast
# CHECK:     kind: type_fn_attr
# CHECK:     default_fn: cast_signed
@linalg_structured_op
def matmul(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
    bfn=BinaryFnAttrDef(default=BinaryFn.mul),
    ufn=UnaryFnAttrDef(default=UnaryFn.exp),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    C[D.m, D.n] += bfn(cast(U, A[D.m, D.k]), cast(U, B[D.k, D.n]))


# CHECK: ---
# CHECK-LABEL: fill
# CHECK: args:
# CHECK:     name: value
# CHECK:     kind: scalar
# CHECK-NOT: shape_map:
# CHECK:     type_var: T
@linalg_structured_op
def fill(value=ScalarDef(T), O=TensorDef(T, S.M, S.K, output=True)):
    O[D.m, D.n] = value


# CHECK: ---
# CHECK-LABEL: strided_copy
# CHECK: args:
# CHECK:     name: I
# CHECK:     kind: input_tensor
# CHECK:     type_var: T
# CHECK:     shape_map: affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0, s1)>
# CHECK:     name: O
# CHECK:     kind: output_tensor
# CHECK:     type_var: T
# CHECK:     shape_map: affine_map<()[s0, s1, s2, s3, s4, s5] -> (s2, s3)>
# CHECK:     name: strides
# CHECK:     kind: index_attr
# CHECK:     index_attr_map: affine_map<()[s0, s1, s2, s3, s4, s5] -> (s4, s5)>
# CHECK:     default_indices:
# CHECK:     - 1
# CHECK:     - 2
@linalg_structured_op
def strided_copy(
    I=TensorDef(T, S.IH, S.IW),
    O=TensorDef(T, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 2]),
):
    O[D.oh, D.ow] = I[D.oh * S.SH, D.ow * S.SW]
