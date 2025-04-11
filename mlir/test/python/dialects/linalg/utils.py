# RUN: %PYTHON %s

from mlir.dialects import arith, func, linalg
from mlir.dialects.linalg.opdsl.lang import *
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


@run
def test_infer_contraction_dimensions_from_ops():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            # === Static shapes ===
            m, n, k = 4, 4, 4
            a_type = RankedTensorType.get((m, k), f32)
            b_type = RankedTensorType.get((k, n), f32)
            c_type = RankedTensorType.get((m, n), f32)

            @func.FuncOp.from_py_func(a_type, b_type, c_type)
            def contraction_fn(a, b, c):
                zero = arith.ConstantOp(value=FloatAttr.get(f32, 0.0), result=f32)
                filled = linalg.fill(zero, outs=[c])
                fill_op = filled.owner

                assert not linalg.isa_contraction_op(zero.operation)
                assert not linalg.isa_contraction_op(fill_op)
                assert linalg.infer_contraction_dimensions(fill_op) is None

                dim_m = AffineDimExpr.get(0)
                dim_n = AffineDimExpr.get(1)
                dim_k = AffineDimExpr.get(2)

                a_map = AffineMap.get(3, 0, [dim_m, dim_k])
                b_map = AffineMap.get(3, 0, [dim_k, dim_n])
                c_map = AffineMap.get(3, 0, [dim_m, dim_n])
                result = linalg.contract(
                    a,
                    b,
                    outs=(filled,),
                    indexing_maps=[a_map, b_map, c_map],
                )
                contraction_op = result.owner

                assert linalg.isa_contraction_op(contraction_op)
                dims = linalg.infer_contraction_dimensions(contraction_op)
                assert dims is not None

                # Expect m=[0], n=[1], k=[2] as per standard matmul
                assert list(dims.m) == [0], f"Expected m=[0], got {list(dims.m)}"
                assert list(dims.n) == [1], f"Expected n=[1], got {list(dims.n)}"
                assert list(dims.k) == [2], f"Expected k=[2], got {list(dims.k)}"
                assert (
                    list(dims.batch) == []
                ), f"Expected batch=[], got {list(dims.batch)}"

            # === Dynamic shape case ===
            dyn = ShapedType.get_dynamic_size()
            a_dyn_type = RankedTensorType.get((4, dyn), f32)
            b_dyn_type = RankedTensorType.get((dyn, 4), f32)
            c_type = RankedTensorType.get((4, 4), f32)

            @func.FuncOp.from_py_func(a_dyn_type, b_dyn_type, c_type)
            def dynamic_contraction_fn(a, b, c):
                zero = arith.ConstantOp(value=FloatAttr.get(f32, 0.0), result=f32)
                filled = linalg.fill(zero, outs=[c])
                dim_m = AffineDimExpr.get(0)
                dim_n = AffineDimExpr.get(1)
                dim_k = AffineDimExpr.get(2)

                a_map = AffineMap.get(3, 0, [dim_m, dim_k])
                b_map = AffineMap.get(3, 0, [dim_k, dim_n])
                c_map = AffineMap.get(3, 0, [dim_m, dim_n])

                result = linalg.contract(
                    a,
                    b,
                    outs=(filled,),
                    indexing_maps=[a_map, b_map, c_map],
                )
                contraction_op = result.owner

                assert linalg.isa_contraction_op(contraction_op)
                dims = linalg.infer_contraction_dimensions(contraction_op)
                assert dims is not None
                assert list(dims.m) == [0], f"Expected m=[0], got {list(dims.m)}"
                assert list(dims.n) == [1], f"Expected n=[1], got {list(dims.n)}"
                assert list(dims.k) == [2], f"Expected k=[2], got {list(dims.k)}"
                assert (
                    list(dims.batch) == []
                ), f"Expected batch=[], got {list(dims.batch)}"
