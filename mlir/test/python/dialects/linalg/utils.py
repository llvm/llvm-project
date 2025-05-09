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

                # Expect m=[0], n=[1], k=[2] as per standard matmul.
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


@run
def test_infer_convolution_dimensions_from_ops():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()

        with InsertionPoint(module.body):
            # === Static shapes ===
            batch, h, w, c_in, kh, kw, c_out = 1, 8, 8, 4, 3, 3, 16
            input_type = RankedTensorType.get((batch, h, w, c_in), f32)
            filter_type = RankedTensorType.get((kh, kw, c_in, c_out), f32)
            output_type = RankedTensorType.get(
                (batch, h - kh + 1, w - kw + 1, c_out), f32
            )

            @func.FuncOp.from_py_func(input_type, filter_type, output_type)
            def conv_fn(input, filter, output):
                zero = arith.ConstantOp(value=FloatAttr.get(f32, 0.0), result=f32)
                filled = linalg.fill(zero, outs=[output])
                fill_op = filled.owner

                assert not linalg.isa_convolution_op(fill_op)
                assert linalg.infer_convolution_dimensions(fill_op) is None

                result = linalg.conv_2d_nhwc_hwcf(input, filter, outs=[filled])
                conv_op = result.owner

                assert linalg.isa_convolution_op(conv_op)
                dims = linalg.infer_convolution_dimensions(conv_op)
                assert dims is not None
                assert list(dims.batch) == [0]
                assert list(dims.output_image) == [1, 2]
                assert list(dims.output_channel) == [3]
                assert list(dims.filter_loop) == [4, 5]
                assert list(dims.input_channel) == [6]
                assert list(dims.depth) == []
                assert list(dims.strides) == [1, 1]
                assert list(dims.dilations) == [1, 1]

            # === Dynamic shapes ===
            dyn = ShapedType.get_dynamic_size()
            dyn_input_type = RankedTensorType.get((batch, dyn, dyn, c_in), f32)
            dyn_output_type = RankedTensorType.get((batch, dyn, dyn, c_out), f32)

            @func.FuncOp.from_py_func(dyn_input_type, filter_type, dyn_output_type)
            def dyn_conv_fn(input, filter, output):
                zero = arith.ConstantOp(value=FloatAttr.get(f32, 0.0), result=f32)
                filled = linalg.fill(zero, outs=[output])
                result = linalg.conv_2d_nhwc_hwcf(input, filter, outs=[filled])
                conv_op = result.owner

                assert linalg.isa_convolution_op(conv_op)
                dims = linalg.infer_convolution_dimensions(conv_op)
                assert dims is not None
                assert list(dims.batch) == [0]
                assert list(dims.output_image) == [1, 2]
                assert list(dims.output_channel) == [3]
                assert list(dims.filter_loop) == [4, 5]
                assert list(dims.input_channel) == [6]
                assert list(dims.depth) == []
                assert list(dims.strides) == [1, 1]
                assert list(dims.dilations) == [1, 1]


@run
def test_get_indexing_maps_attr():
    with Context(), Location.unknown():
        module = Module.create()
        f32 = F32Type.get()
        with InsertionPoint(module.body):
            a_type = RankedTensorType.get((4, 8), f32)
            b_type = RankedTensorType.get((8, 16), f32)
            c_type = RankedTensorType.get((4, 16), f32)

            dim_m = AffineDimExpr.get(0)
            dim_n = AffineDimExpr.get(1)
            dim_k = AffineDimExpr.get(2)

            a_map = AffineMap.get(3, 0, [dim_m, dim_k])
            b_map = AffineMap.get(3, 0, [dim_k, dim_n])
            c_map = AffineMap.get(3, 0, [dim_m, dim_n])

            @func.FuncOp.from_py_func(a_type, b_type, c_type)
            def matmul_func(a, b, c):
                zero = arith.ConstantOp(value=FloatAttr.get(f32, 0.0), result=f32)
                assert not linalg.get_indexing_maps(
                    zero.operation
                ), "Expected no indexing_maps on non-linalg op"

                init = linalg.fill(zero, outs=[c])
                fill_op = init.owner
                fill_maps = linalg.get_indexing_maps(fill_op)
                assert fill_maps is not None
                assert len(fill_maps) == 2

                # The fill op should have maps like (d0, d1) -> () and (d0, d1).
                fill_input_map = fill_maps[0].value
                fill_output_map = fill_maps[1].value
                assert fill_input_map == AffineMap.get(2, 0, [])
                assert fill_output_map == AffineMap.get(2, 0, [dim_m, dim_n])

                result = linalg.matmul(a, b, outs=(init,))
                matmul_op = result.owner
                matmul_maps = linalg.get_indexing_maps(matmul_op)
                assert matmul_maps is not None
                assert len(matmul_maps) == 3

                maps = [map_attr.value for map_attr in matmul_maps]
                assert maps[0] == a_map
                assert maps[1] == b_map
                assert maps[2] == c_map
