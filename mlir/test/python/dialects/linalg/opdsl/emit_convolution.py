# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import linalg

from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2


@linalg_structured_op
def conv_poly(
    I=TensorDef(T1, S.N, S.IH, S.IW, S.C),
    K=TensorDef(T2, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=IndexAttrDef(S.SH, S.SW, default=[1, 1]),
    dilations=IndexAttrDef(S.DH, S.DW, default=[1, 2])):
  domain(D.n, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.c] += TypeFn.cast_signed(
      U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW,
           D.c]) * TypeFn.cast_signed(U, K[D.kh, D.kw, D.c])


with Context() as ctx, Location.unknown():
  module = Module.create()
  f32 = F32Type.get()
  i32 = IntegerType.get_signless(32)
  with InsertionPoint(module.body):

    # Convolution indexing maps.
    # CHECK: #[[$CONV_MAP_I:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3, d2 * 4 + d4 * 2, d5)>
    # CHECK: #[[$CONV_MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
    # CHECK: #[[$CONV_MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

    # CHECK-LABEL: @test_f32i32_conv
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$CONV_MAP_I]], #[[$CONV_MAP_K]], #[[$CONV_MAP_O]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[FILTER:.+]]: f32, %[[OUT:.+]]: i32)
    # CHECK-NEXT:   %[[IN_CAST:.+]] = arith.fptosi %[[IN:.+]] : f32 to i32
    # CHECK-NEXT:   %[[FILTER_CAST:.+]] = arith.fptosi %[[FILTER:.+]] : f32 to i32
    # CHECK-NEXT:   %[[PROD:.+]] = arith.muli %[[IN_CAST]], %[[FILTER_CAST]] : i32
    # CHECK-NEXT:   %[[SUM:.+]] = arith.addi %[[OUT]], %[[PROD]] : i32
    # CHECK-NEXT:   linalg.yield %[[SUM]] : i32
    # CHECK-NEXT: -> tensor<1x2x4x1xi32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2, 1), f32),
        RankedTensorType.get((1, 2, 4, 1), i32))
    def test_f32i32_conv(input, filter, init_result):
      # Use default dilations and set non-default strides.
      return conv_poly(
          input, filter, outs=[init_result], strides=[2, 4])


print(module)
