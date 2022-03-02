# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import linalg

from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2


@linalg_structured_op
def fill_poly(value=ScalarDef(T1), O=TensorDef(U, output=True)):
  O[None] = TypeFn.cast_signed(U, value)


with Context() as ctx, Location.unknown():
  module = Module.create()
  f32 = F32Type.get()
  with InsertionPoint(module.body):

    # Fill indexing maps.
    # CHECK-DAG: #[[$MAP0:.+]] = affine_map<() -> ()>
    # CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> ()>
    # CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>

    # CHECK-LABEL: @test_fill_0d
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP0]]
    # CHECK-SAME: iterator_types = []
    @builtin.FuncOp.from_py_func(f32, RankedTensorType.get([], f32))
    def test_fill_0d(value, init_result):
      return fill_poly(value, outs=[init_result])

    # CHECK-LABEL: @test_fill_2d
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$MAP1]], #[[$MAP2]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel"]
    @builtin.FuncOp.from_py_func(f32, RankedTensorType.get([4, 16], f32))
    def test_fill_2d(value, init_result):
      return fill_poly(value, outs=[init_result])


print(module)
