# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import arith
from mlir.dialects import builtin
from mlir.dialects import func


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: TEST: testConstantOp


@constructAndPrintInModule
def testConstantOp():
    c1 = arith.ConstantOp(IntegerType.get_signless(32), 42)
    c2 = arith.ConstantOp(IntegerType.get_signless(64), 100)
    c3 = arith.ConstantOp(F32Type.get(), 3.14)
    c4 = arith.ConstantOp(F64Type.get(), 1.23)
    # CHECK: 42
    print(c1.literal_value)

    # CHECK: 100
    print(c2.literal_value)

    # CHECK: 3.140000104904175
    print(c3.literal_value)

    # CHECK: 1.23
    print(c4.literal_value)


# CHECK: = arith.constant 42 : i32
# CHECK: = arith.constant 100 : i64
# CHECK: = arith.constant 3.140000e+00 : f32
# CHECK: = arith.constant 1.230000e+00 : f64


# CHECK-LABEL: TEST: testVectorConstantOp
@constructAndPrintInModule
def testVectorConstantOp():
    int_type = IntegerType.get_signless(32)
    vec_type = VectorType.get([2, 2], int_type)
    c1 = arith.ConstantOp(
        vec_type, DenseElementsAttr.get_splat(vec_type, IntegerAttr.get(int_type, 42))
    )
    try:
        print(c1.literal_value)
    except ValueError as e:
        assert "only integer and float constants have literal values" in str(e)
    else:
        assert False


# CHECK: = arith.constant dense<42> : vector<2x2xi32>


# CHECK-LABEL: TEST: testConstantIndexOp
@constructAndPrintInModule
def testConstantIndexOp():
    c1 = arith.ConstantOp.create_index(10)
    # CHECK: 10
    print(c1.literal_value)


# CHECK: = arith.constant 10 : index


# CHECK-LABEL: TEST: testFunctionCalls
@constructAndPrintInModule
def testFunctionCalls():
    foo = func.FuncOp("foo", ([], []))
    foo.sym_visibility = StringAttr.get("private")
    bar = func.FuncOp("bar", ([], [IndexType.get()]))
    bar.sym_visibility = StringAttr.get("private")
    qux = func.FuncOp("qux", ([], [F32Type.get()]))
    qux.sym_visibility = StringAttr.get("private")

    con = func.ConstantOp(qux.type, qux.sym_name.value)
    assert con.type == qux.type

    with InsertionPoint(func.FuncOp("caller", ([], [])).add_entry_block()):
        func.CallOp(foo, [])
        func.CallOp([IndexType.get()], "bar", [])
        func.CallOp([F32Type.get()], FlatSymbolRefAttr.get("qux"), [])
        func.ReturnOp([])


# CHECK: func private @foo()
# CHECK: func private @bar() -> index
# CHECK: func private @qux() -> f32
# CHECK: %f = func.constant @qux : () -> f32
# CHECK: func @caller() {
# CHECK:   call @foo() : () -> ()
# CHECK:   %0 = call @bar() : () -> index
# CHECK:   %1 = call @qux() : () -> f32
# CHECK:   return
# CHECK: }


# CHECK-LABEL: TEST: testFunctionArgAttrs
@constructAndPrintInModule
def testFunctionArgAttrs():
    foo = func.FuncOp("foo", ([F32Type.get()], []))
    foo.sym_visibility = StringAttr.get("private")
    foo2 = func.FuncOp("foo2", ([F32Type.get(), F32Type.get()], []))
    foo2.sym_visibility = StringAttr.get("private")

    empty_attr = DictAttr.get({})
    test_attr = DictAttr.get({"test.foo": StringAttr.get("bar")})
    test_attr2 = DictAttr.get({"test.baz": StringAttr.get("qux")})

    assert len(foo.arg_attrs) == 1
    assert foo.arg_attrs[0] == empty_attr

    foo.arg_attrs = [test_attr]
    assert foo.arg_attrs[0]["test.foo"] == StringAttr.get("bar")

    assert len(foo2.arg_attrs) == 2
    assert foo2.arg_attrs == ArrayAttr.get([empty_attr, empty_attr])

    foo2.arg_attrs = [empty_attr, test_attr2]
    assert foo2.arg_attrs == ArrayAttr.get([empty_attr, test_attr2])


# CHECK: func private @foo(f32 {test.foo = "bar"})
# CHECK: func private @foo2(f32, f32  {test.baz = "qux"})
