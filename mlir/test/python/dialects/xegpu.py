# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import func, xegpu


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


# CHECK-LABEL: TEST: testParseDpas
@run
def testParseDpas():
    module = Module.parse(
        """
        func.func @dpas(%a: vector<8x16xf16>, %b: vector<16x16xf16>,
                        %c: vector<8x16xf32>) -> vector<8x16xf32> {
          %0 = xegpu.dpas %a, %b
            : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
          %1 = xegpu.dpas %a, %b, %c
            : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
          return %1 : vector<8x16xf32>
        }
        """
    )
    assert module.operation.verify()
    function = module.body.operations[0]
    dpas, dpas_acc, return_op = function.body.blocks[0].operations
    assert isinstance(dpas, xegpu.DpasOp)
    assert isinstance(dpas_acc, xegpu.DpasOp)
    assert not isinstance(return_op, xegpu.DpasOp)
    assert dpas.acc is None
    assert dpas_acc.acc == function.arguments[2]
    assert dpas.lhs == function.arguments[0]
    assert dpas.rhs == function.arguments[1]
    print(module)
    # CHECK: xegpu.dpas %arg0, %arg1 :
    # CHECK: xegpu.dpas %arg0, %arg1, %arg2 :


# CHECK-LABEL: TEST: testParseLoadNd
@run
def testParseLoadNd():
    module = Module.parse(
        """
        func.func @load_nd(%desc: !xegpu.tensor_desc<8x16xf32>) -> vector<8x16xf32> {
          %0 = xegpu.load_nd %desc[0, 0]
            : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
          return %0 : vector<8x16xf32>
        }
        """
    )
    assert module.operation.verify()
    function = module.body.operations[0]
    load = function.body.blocks[0].operations[0]
    assert isinstance(load, xegpu.LoadNdOp)
    assert load.TensorDesc == function.arguments[0]
    assert len(load.offsets) == 0
    assert list(load.const_offsets) == [0, 0]
    assert load.value.type == VectorType.get([8, 16], F32Type.get())
    print(module)
    # CHECK: xegpu.load_nd %arg0[0, 0] : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>


# CHECK-LABEL: TEST: testBuildDpas
@run
def testBuildDpas():
    module = Module.create()
    a_type = VectorType.get([8, 16], F16Type.get())
    b_type = VectorType.get([16, 16], F16Type.get())
    c_type = VectorType.get([8, 16], F32Type.get())
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(a_type, b_type, c_type)
        def dpas(a, b, c):
            product = xegpu.DpasOp(c_type, a, b)
            result = xegpu.DpasOp(c_type, a, b, acc=c)
            assert isinstance(product, xegpu.DpasOp)
            assert isinstance(result.operation.opview, xegpu.DpasOp)
            assert product.acc is None
            assert result.acc == c
            assert result.result.type == c_type
            return result

    assert module.operation.verify()
    print(module)
    # CHECK: xegpu.dpas %arg0, %arg1 :
    # CHECK: xegpu.dpas %arg0, %arg1, %arg2 :


# CHECK-LABEL: TEST: testBuildLoadNd
@run
def testBuildLoadNd():
    module = Module.create()
    desc_type = Type.parse("!xegpu.tensor_desc<8x16xf32>")
    value_type = VectorType.get([8, 16], F32Type.get())
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(desc_type)
        def load_nd(desc):
            load = xegpu.LoadNdOp(value_type, desc, [], [0, 0])
            assert isinstance(load.operation.opview, xegpu.LoadNdOp)
            assert load.TensorDesc == desc
            assert len(load.offsets) == 0
            assert list(load.const_offsets) == [0, 0]
            assert load.value.type == value_type
            return load

    assert module.operation.verify()
    print(module)
    # CHECK: xegpu.load_nd %arg0[0, 0] : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
