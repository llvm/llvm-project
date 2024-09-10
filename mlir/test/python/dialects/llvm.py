# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import llvm


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testStructType
@constructAndPrintInModule
def testStructType():
    print(llvm.StructType.get_literal([]))
    # CHECK: !llvm.struct<()>

    i8, i32, i64 = tuple(map(lambda x: IntegerType.get_signless(x), [8, 32, 64]))
    print(llvm.StructType.get_literal([i8, i32, i64]))
    print(llvm.StructType.get_literal([i32]))
    print(llvm.StructType.get_literal([i32, i32], packed=True))
    literal = llvm.StructType.get_literal([i8, i32, i64])
    assert len(literal.body) == 3
    print(*tuple(literal.body))
    assert literal.name is None
    # CHECK: !llvm.struct<(i8, i32, i64)>
    # CHECK: !llvm.struct<(i32)>
    # CHECK: !llvm.struct<packed (i32, i32)>
    # CHECK: i8 i32 i64

    assert llvm.StructType.get_literal([i32]) == llvm.StructType.get_literal([i32])
    assert llvm.StructType.get_literal([i32]) != llvm.StructType.get_literal([i64])

    print(llvm.StructType.get_identified("foo"))
    print(llvm.StructType.get_identified("bar"))
    # CHECK: !llvm.struct<"foo", opaque>
    # CHECK: !llvm.struct<"bar", opaque>

    assert llvm.StructType.get_identified("foo") == llvm.StructType.get_identified(
        "foo"
    )
    assert llvm.StructType.get_identified("foo") != llvm.StructType.get_identified(
        "bar"
    )

    foo_struct = llvm.StructType.get_identified("foo")
    print(foo_struct.name)
    print(foo_struct.body)
    assert foo_struct.opaque
    foo_struct.set_body([i32, i64])
    print(*tuple(foo_struct.body))
    print(foo_struct)
    assert not foo_struct.packed
    assert not foo_struct.opaque
    assert llvm.StructType.get_identified("foo") == foo_struct
    # CHECK: foo
    # CHECK: None
    # CHECK: i32 i64
    # CHECK: !llvm.struct<"foo", (i32, i64)>

    bar_struct = llvm.StructType.get_identified("bar")
    bar_struct.set_body([i32], packed=True)
    print(bar_struct)
    assert bar_struct.packed
    # CHECK: !llvm.struct<"bar", packed (i32)>

    # Same body, should not raise.
    foo_struct.set_body([i32, i64])

    try:
        foo_struct.set_body([])
    except ValueError as e:
        pass
    else:
        assert False, "expected exception not raised"

    try:
        bar_struct.set_body([i32])
    except ValueError as e:
        pass
    else:
        assert False, "expected exception not raised"

    print(llvm.StructType.new_identified("foo", []))
    assert llvm.StructType.new_identified("foo", []) != llvm.StructType.new_identified(
        "foo", []
    )
    # CHECK: !llvm.struct<"foo{{[^"]+}}

    opaque = llvm.StructType.get_opaque("opaque")
    print(opaque)
    assert opaque.opaque
    # CHECK: !llvm.struct<"opaque", opaque>


# CHECK-LABEL: testSmoke
@constructAndPrintInModule
def testSmoke():
    mat64f32_t = Type.parse(
        "!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>"
    )
    result = llvm.UndefOp(mat64f32_t)
    # CHECK: %0 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>


# CHECK-LABEL: testPointerType
@constructAndPrintInModule
def testPointerType():
    ptr = llvm.PointerType.get()
    # CHECK: !llvm.ptr
    print(ptr)

    ptr_with_addr = llvm.PointerType.get(1)
    # CHECK: !llvm.ptr<1>
    print(ptr_with_addr)


# CHECK-LABEL: testConstant
@constructAndPrintInModule
def testConstant():
    i32 = IntegerType.get_signless(32)
    c_128 = llvm.mlir_constant(IntegerAttr.get(i32, 128))
    # CHECK: %{{.*}} = llvm.mlir.constant(128 : i32) : i32
    print(c_128.owner)


# CHECK-LABEL: testIntrinsics
@constructAndPrintInModule
def testIntrinsics():
    i32 = IntegerType.get_signless(32)
    ptr = llvm.PointerType.get()
    c_128 = llvm.mlir_constant(IntegerAttr.get(i32, 128))
    # CHECK: %[[CST128:.*]] = llvm.mlir.constant(128 : i32) : i32
    print(c_128.owner)

    alloca = llvm.alloca(ptr, c_128, i32)
    # CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[CST128]] x i32 : (i32) -> !llvm.ptr
    print(alloca.owner)

    c_0 = llvm.mlir_constant(IntegerAttr.get(IntegerType.get_signless(8), 0))
    # CHECK: %[[CST0:.+]] = llvm.mlir.constant(0 : i8) : i8
    print(c_0.owner)

    result = llvm.intr_memset(alloca, c_0, c_128, False)
    # CHECK: "llvm.intr.memset"(%[[ALLOCA]], %[[CST0]], %[[CST128]]) <{isVolatile = false}> : (!llvm.ptr, i8, i32) -> ()
    print(result)
