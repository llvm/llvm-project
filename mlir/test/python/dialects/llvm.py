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

    typ = Type.parse('!llvm.struct<"zoo", (i32, i64)>')
    assert isinstance(typ, llvm.StructType)


# CHECK-LABEL: testArrayType
@constructAndPrintInModule
def testArrayType():
    i32 = IntegerType.get_signless(32)
    i8 = IntegerType.get_signless(8)

    arr = llvm.ArrayType.get(i32, 4)
    # CHECK: !llvm.array<4 x i32>
    print(arr)
    assert arr.element_type == i32
    assert arr.num_elements == 4

    arr2 = llvm.ArrayType.get(i8, 12)
    # CHECK: !llvm.array<12 x i8>
    print(arr2)
    assert arr2.element_type == i8
    assert arr2.num_elements == 12

    typ = Type.parse("!llvm.array<4 x i32>")
    assert isinstance(typ, llvm.ArrayType)
    assert typ == arr


# CHECK-LABEL: testArrayTypeOps
@constructAndPrintInModule
def testArrayTypeOps():
    i32 = IntegerType.get_signless(32)
    arr_t = llvm.ArrayType.get(i32, 4)

    undef = llvm.UndefOp(arr_t)
    c_42 = llvm.mlir_constant(IntegerAttr.get(i32, 42))
    inserted = llvm.insertvalue(undef, c_42, [0])
    llvm.extractvalue(i32, inserted, [0])

    # CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
    # CHECK: %[[C42:.*]] = llvm.mlir.constant(42 : i32) : i32
    # CHECK: %[[INS:.*]] = llvm.insertvalue %[[C42]], %[[UNDEF]][0] : !llvm.array<4 x i32>
    # CHECK: %{{.*}} = llvm.extractvalue %[[INS]][0] : !llvm.array<4 x i32>


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

    typ = Type.parse("!llvm.ptr<1>")
    assert isinstance(typ, llvm.PointerType)


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


# CHECK-LABEL: testTranslateToLLVMIR
@constructAndPrintInModule
def testTranslateToLLVMIR():
    with Context(), Location.unknown():
        module = Module.parse(
            """\
            llvm.func @add(%arg0: i64, %arg1: i64) -> i64 { 
               %0 = llvm.add %arg0, %arg1  : i64 
               llvm.return %0 : i64 
            }
        """
        )
        # CHECK: define i64 @add(i64 %0, i64 %1) {
        # CHECK:   %3 = add i64 %0, %1
        # CHECK:   ret i64 %3
        # CHECK: }
        print(llvm.translate_module_to_llvmir(module.operation))


# CHECK-LABEL: testMetadataAttrs
@constructAndPrintInModule
def testMetadataAttrs():
    # MDStringAttr
    md_str = llvm.MDStringAttr.get("foo.buffer")
    # CHECK: #llvm.md_string<"foo.buffer">
    print(md_str)
    assert md_str.value == "foo.buffer"

    # MDConstantAttr
    i32 = IntegerType.get_signless(32)
    md_const = llvm.MDConstantAttr.get(IntegerAttr.get(i32, 42))
    # CHECK: #llvm.md_const<42 : i32>
    print(md_const)

    # MDFuncAttr
    md_func = llvm.MDFuncAttr.get("my_kernel")
    # CHECK: #llvm.md_func<@my_kernel>
    print(md_func)
    assert md_func.name == "my_kernel"

    # MDNodeAttr - empty
    md_empty = llvm.MDNodeAttr.get([])
    # CHECK: #llvm.md_node<>
    print(md_empty)
    assert len(md_empty) == 0

    # MDNodeAttr - with operands
    md_node = llvm.MDNodeAttr.get([md_const, md_str])
    # CHECK: #llvm.md_node<#llvm.md_const<42 : i32>, #llvm.md_string<"foo.buffer">>
    print(md_node)
    assert len(md_node) == 2

    # MDNodeAttr - __getitem__
    # CHECK: #llvm.md_const<42 : i32>
    print(md_node[0])
    # CHECK: #llvm.md_string<"foo.buffer">
    print(md_node[1])
    assert str(md_node[0]) == str(md_const)
    assert str(md_node[1]) == str(md_str)

    # MDNodeAttr - nested
    md_nested = llvm.MDNodeAttr.get([md_node, md_empty])
    # CHECK: #llvm.md_node<#llvm.md_node<#llvm.md_const<42 : i32>, #llvm.md_string<"foo.buffer">>, #llvm.md_node<>>
    print(md_nested)
    assert len(md_nested) == 2


# CHECK-LABEL: testNamedMetadata
@constructAndPrintInModule
def testNamedMetadata():
    void = Type.parse("!llvm.void")
    func_ty = llvm.FunctionType.get(void, [])

    llvm.LLVMFuncOp("my_kernel", TypeAttr.get(func_ty))
    # CHECK-LABEL:   llvm.func @my_kernel()

    llvm.NamedMetadataOp(
        metadata_name="foo.version",
        nodes=ArrayAttr.get(
            [
                llvm.MDNodeAttr.get(
                    [llvm.md_const(1), llvm.md_const(0), llvm.md_const(0)]
                )
            ]
        ),
    )
    # CHECK: llvm.named_metadata "foo.version" [#llvm.md_node<#llvm.md_const<1 : i32>, #llvm.md_const<0 : i32>, #llvm.md_const<0 : i32>>]

    llvm.NamedMetadataOp(
        metadata_name="foo.language_version",
        nodes=ArrayAttr.get(
            [
                llvm.MDNodeAttr.get(
                    [
                        llvm.md_str("Bar"),
                        llvm.md_const(1),
                        llvm.md_const(2),
                        llvm.md_const(3),
                    ]
                )
            ]
        ),
    )
    # CHECK: llvm.named_metadata "foo.language_version" [#llvm.md_node<#llvm.md_string<"Bar">, #llvm.md_const<1 : i32>, #llvm.md_const<2 : i32>, #llvm.md_const<3 : i32>>]

    buf0 = llvm.MDNodeAttr.get(
        [
            llvm.md_const(0),
            llvm.md_str("foo.buffer"),
            llvm.md_str("foo.idx"),
            llvm.md_const(0),
            llvm.md_const(1),
            llvm.md_str("foo.read"),
            llvm.md_str("foo.address_space"),
            llvm.md_const(1),
            llvm.md_str("foo.size"),
            llvm.md_const(4),
            llvm.md_str("foo.align_size"),
            llvm.md_const(4),
        ]
    )

    llvm.NamedMetadataOp(
        metadata_name="foo.kernel",
        nodes=ArrayAttr.get(
            [
                llvm.MDNodeAttr.get(
                    [
                        llvm.MDFuncAttr.get("my_kernel"),
                        llvm.MDNodeAttr.get([]),
                        buf0,
                    ]
                )
            ]
        ),
    )
    # CHECK:       llvm.named_metadata "foo.kernel" [
    # CHECK-SAME:  #llvm.md_node<
    # CHECK-SAME:      #llvm.md_func<@my_kernel>,
    # CHECK-SAME:      #llvm.md_node<>,
    # CHECK-SAME:      #llvm.md_node<
    # CHECK-SAME:          #llvm.md_const<0 : i32>,
    # CHECK-SAME:          #llvm.md_string<"foo.buffer">,
    # CHECK-SAME:          #llvm.md_string<"foo.idx">,
    # CHECK-SAME:          #llvm.md_const<0 : i32>,
    # CHECK-SAME:          #llvm.md_const<1 : i32>,
    # CHECK-SAME:          #llvm.md_string<"foo.read">,
    # CHECK-SAME:          #llvm.md_string<"foo.address_space">,
    # CHECK-SAME:          #llvm.md_const<1 : i32>,
    # CHECK-SAME:          #llvm.md_string<"foo.size">,
    # CHECK-SAME:          #llvm.md_const<4 : i32>,
    # CHECK-SAME:          #llvm.md_string<"foo.align_size">,
    # CHECK-SAME:          #llvm.md_const<4 : i32>>
    # CHECK-SAME:    >]
