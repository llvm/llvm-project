# RUN: %PYTHON %s | FileCheck %s

import gc
import io
from tempfile import NamedTemporaryFile
from mlir.ir import *
from mlir.dialects.builtin import ModuleOp
from mlir.dialects import arith, func, scf, shape
from mlir.dialects._ods_common import _cext
from mlir.extras import types as T


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


def expect_index_error(callback):
    try:
        _ = callback()
        raise RuntimeError("Expected IndexError")
    except IndexError:
        pass


# Verify iterator based traversal of the op/region/block hierarchy.
# CHECK-LABEL: TEST: testTraverseOpRegionBlockIterators
@run
def testTraverseOpRegionBlockIterators():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    func.func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """,
        ctx,
    )
    op = module.operation
    assert op.context is ctx
    # Note, __nb_signature__ stores the fully-qualified signature - the actual type stub emitted is
    # class RegionSequence(Sequence[Region])
    # CHECK: class RegionSequence(collections.abc.Sequence[mlir._mlir_libs._mlir.ir.Region])
    print(RegionSequence.__nb_signature__)
    # Get the block using iterators off of the named collections.
    regions = list(op.regions[:])
    blocks = list(regions[0].blocks)
    # CHECK: MODULE REGIONS=1 BLOCKS=1
    print(f"MODULE REGIONS={len(regions)} BLOCKS={len(blocks)}")

    # Should verify.
    # CHECK: .verify = True
    print(f".verify = {module.operation.verify()}")

    # Get the blocks from the default collection.
    default_blocks = list(regions[0])
    # They should compare equal regardless of how obtained.
    assert default_blocks == blocks

    # Should be able to get the operations from either the named collection
    # or the block.
    operations = list(blocks[0].operations)
    default_operations = list(blocks[0])
    assert default_operations == operations

    def walk_operations(indent, op):
        for i, region in enumerate(op.regions):
            print(f"{indent}REGION {i}:")
            for j, block in enumerate(region):
                print(f"{indent}  BLOCK {j}:")
                for k, child_op in enumerate(block):
                    print(f"{indent}    OP {k}: {child_op}")
                    walk_operations(indent + "      ", child_op)

    # CHECK: REGION 0:
    # CHECK:   BLOCK 0:
    # CHECK:     OP 0: func
    # CHECK:       REGION 0:
    # CHECK:         BLOCK 0:
    # CHECK:           OP 0: %0 = "custom.addi"
    # CHECK:           OP 1: func.return
    walk_operations("", op)

    # CHECK:    Region iter: <iterator
    # CHECK:     Block iter: <mlir.{{.+}}.BlockIterator
    # CHECK: Operation iter: <mlir.{{.+}}.OperationIterator
    print("   Region iter:", iter(op.regions))
    print("    Block iter:", iter(op.regions[-1]))
    print("Operation iter:", iter(op.regions[-1].blocks[-1]))

    try:
        op.regions[-42]
    except IndexError as e:
        # CHECK: Region OOB: index out of range
        print("Region OOB:", e)
    try:
        op.regions[0].blocks[-42]
    except IndexError as e:
        # CHECK: attempt to access out of bounds block
        print(e)
    try:
        op.regions[0].blocks[0].operations[-42]
    except IndexError as e:
        # CHECK: attempt to access out of bounds operation
        print(e)

    # Verify that iterating a sliced region list yields the correct
    # number of elements (i.e. respects length and step).
    with Location.unknown(ctx):
        op4 = Operation.create("custom.op", regions=4)
        r = op4.regions
        assert len(list(r[:])) == 4
        assert len(list(r[1:])) == 3
        assert len(list(r[::2])) == 2
        assert len(list(r[1:3])) == 2
        assert len(list(r[::-1])) == 4


# Verify index based traversal of the op/region/block hierarchy.
# CHECK-LABEL: TEST: testTraverseOpRegionBlockIndices
@run
def testTraverseOpRegionBlockIndices():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    func.func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """,
        ctx,
    )

    def walk_operations(indent, op):
        for i in range(len(op.regions)):
            region = op.regions[i]
            print(f"{indent}REGION {i}:")
            for j in range(len(region.blocks)):
                block = region.blocks[j]
                print(f"{indent}  BLOCK {j}:")
                for k in range(len(block.operations)):
                    child_op = block.operations[k]
                    print(f"{indent}    OP {k}: {child_op}")
                    print(
                        f"{indent}    OP {k}: parent {child_op.operation.parent.name}"
                    )
                    walk_operations(indent + "      ", child_op)

    # CHECK: REGION 0:
    # CHECK:   BLOCK 0:
    # CHECK:     OP 0: func
    # CHECK:     OP 0: parent builtin.module
    # CHECK:       REGION 0:
    # CHECK:         BLOCK 0:
    # CHECK:           OP 0: %0 = "custom.addi"
    # CHECK:           OP 0: parent func.func
    # CHECK:           OP 1: func.return
    # CHECK:           OP 1: parent func.func
    walk_operations("", module.operation)


# CHECK-LABEL: TEST: testBlockAndRegionOwners
@run
def testBlockAndRegionOwners():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    builtin.module {
      func.func @f() {
        func.return
      }
    }
  """,
        ctx,
    )

    assert module.operation.regions[0].owner == module.operation
    assert module.operation.regions[0].blocks[0].owner == module.operation

    func = module.body.operations[0]
    assert func.operation.regions[0].owner == func
    assert func.operation.regions[0].blocks[0].owner == func


# CHECK-LABEL: TEST: testBlockArgumentList
@run
def testBlockArgumentList():
    with Context() as ctx:
        module = Module.parse(
            r"""
      func.func @f1(%arg0: i32, %arg1: f64, %arg2: index) {
        return
      }
    """,
            ctx,
        )
        func_op = module.body.operations[0]
        entry_block = func_op.regions[0].blocks[0]
        assert len(entry_block.arguments) == 3
        # CHECK: Argument 0, type i32
        # CHECK: Argument 1, type f64
        # CHECK: Argument 2, type index
        for arg in entry_block.arguments:
            print(f"Argument {arg.arg_number}, type {arg.type}")
            new_type = IntegerType.get_signless(8 * (arg.arg_number + 1))
            arg.set_type(new_type)

        # CHECK: Argument 0, type i8
        # CHECK: Argument 1, type i16
        # CHECK: Argument 2, type i24
        for arg in entry_block.arguments:
            print(f"Argument {arg.arg_number}, type {arg.type}")

        # CHECK: Matched Arg 0, type i8
        # CHECK: Matched Arg 1, type i16
        # CHECK: Matched Arg 2, type i24
        match func_op:
            case func.FuncOp(body=Region(blocks=[Block(arguments=[a0, a1])])):
                assert False
            case func.FuncOp(body=Region(blocks=[Block(arguments=[a0, a1, a2, a3])])):
                assert False
            case func.FuncOp(body=Region(blocks=[Block(arguments=[a0, a1, a2])])):
                print(f"Matched Arg 0, type {a0.type}")
                print(f"Matched Arg 1, type {a1.type}")
                print(f"Matched Arg 2, type {a2.type}")
            case _:
                assert False

        # Check that slicing works for block argument lists.
        # CHECK: Argument 1, type i16
        # CHECK: Argument 2, type i24
        for arg in entry_block.arguments[1:]:
            print(f"Argument {arg.arg_number}, type {arg.type}")

        # Check that we can concatenate slices of argument lists.
        # CHECK: Length: 4
        print("Length: ", len(entry_block.arguments[:2] + entry_block.arguments[1:]))

        # CHECK: Type: i8
        # CHECK: Type: i16
        # CHECK: Type: i24
        for t in entry_block.arguments.types:
            print("Type: ", t)

        # Check that slicing and type access compose.
        # CHECK: Sliced type: i16
        # CHECK: Sliced type: i24
        for t in entry_block.arguments[1:].types:
            print("Sliced type: ", t)

        # Check that slice addition works as expected.
        # CHECK: Argument 2, type i24
        # CHECK: Argument 0, type i8
        restructured = entry_block.arguments[-1:] + entry_block.arguments[:1]
        for arg in restructured:
            print(f"Argument {arg.arg_number}, type {arg.type}")


# CHECK-LABEL: TEST: testOperationOperands
@run
def testOperationOperands():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(
            r"""
      func.func @f1(%arg0: i32) {
        %0 = "test.producer"() : () -> i64
        "test.consumer"(%arg0, %0) : (i32, i64) -> ()
        return
      }"""
        )
        func_op = module.body.operations[0]
        entry_block = func_op.regions[0].blocks[0]
        consumer = entry_block.operations[1]
        assert len(consumer.operands) == 2
        # CHECK: Operand 0, type i32
        # CHECK: Operand 1, type i64
        for i, operand in enumerate(consumer.operands):
            print(f"Operand {i}, type {operand.type}")

        match module.body.operations:
            case [
                func.FuncOp(
                    body=Region(
                        blocks=[
                            Block(
                                operations=[
                                    _,
                                    OpView(operands=[o1, o2]) as matched_consumer,
                                    *_,
                                ],
                            ),
                        ],
                    ),
                ),
            ]:
                print(f"Matched Operand 0, type {o1.type}")
                print(f"Matched Operand 1, type {o2.type}")


# CHECK-LABEL: TEST: testOperationOperandsSlice
@run
def testOperationOperandsSlice():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(
            r"""
      func.func @f1() {
        %0 = "test.producer0"() : () -> i64
        %1 = "test.producer1"() : () -> i64
        %2 = "test.producer2"() : () -> i64
        %3 = "test.producer3"() : () -> i64
        %4 = "test.producer4"() : () -> i64
        "test.consumer"(%0, %1, %2, %3, %4) : (i64, i64, i64, i64, i64) -> ()
        return
      }"""
        )
        func = module.body.operations[0]
        entry_block = func.regions[0].blocks[0]
        consumer = entry_block.operations[5]
        assert len(consumer.operands) == 5
        for left, right in zip(consumer.operands, consumer.operands[::-1][::-1]):
            assert left == right

        # CHECK: test.producer0
        # CHECK: test.producer1
        # CHECK: test.producer2
        # CHECK: test.producer3
        # CHECK: test.producer4
        full_slice = consumer.operands[:]
        for operand in full_slice:
            print(operand)

        # CHECK: test.producer0
        # CHECK: test.producer1
        first_two = consumer.operands[0:2]
        for operand in first_two:
            print(operand)

        # CHECK: test.producer3
        # CHECK: test.producer4
        last_two = consumer.operands[3:]
        for operand in last_two:
            print(operand)

        # CHECK: test.producer0
        # CHECK: test.producer2
        # CHECK: test.producer4
        even = consumer.operands[::2]
        for operand in even:
            print(operand)

        # CHECK: test.producer2
        fourth = consumer.operands[::2][1::2]
        for operand in fourth:
            print(operand)


# CHECK-LABEL: TEST: testOperationOperandsSet
@run
def testOperationOperandsSet():
    with Context() as ctx, Location.unknown(ctx):
        ctx.allow_unregistered_dialects = True
        module = Module.parse(
            r"""
      func.func @f1() {
        %0 = "test.producer0"() : () -> i64
        %1 = "test.producer1"() : () -> i64
        %2 = "test.producer2"() : () -> i64
        "test.consumer"(%0) : (i64) -> ()
        return
      }"""
        )
        func = module.body.operations[0]
        entry_block = func.regions[0].blocks[0]
        producer1 = entry_block.operations[1]
        producer2 = entry_block.operations[2]
        consumer = entry_block.operations[3]
        assert len(consumer.operands) == 1
        type = consumer.operands[0].type

        # CHECK: test.producer1
        consumer.operands[0] = producer1.result
        print(consumer.operands[0])

        # CHECK: test.producer2
        consumer.operands[-1] = producer2.result
        print(consumer.operands[0])


# CHECK-LABEL: TEST: testDetachedOperation
@run
def testDetachedOperation():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signed(32)
        op1 = Operation.create(
            "custom.op1",
            results=[i32, i32],
            regions=1,
            attributes={
                "foo": StringAttr.get("foo_value"),
                "bar": StringAttr.get("bar_value"),
            },
        )
        # CHECK: %0:2 = "custom.op1"() ({
        # CHECK: }) {bar = "bar_value", foo = "foo_value"} : () -> (si32, si32)
        print(op1)

    # TODO: Check successors once enough infra exists to do it properly.


# CHECK-LABEL: TEST: testOperationInsertionPoint
@run
def testOperationInsertionPoint():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    func.func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """,
        ctx,
    )

    # Create test op.
    with Location.unknown(ctx):
        op1 = Operation.create("custom.op1")
        op2 = Operation.create("custom.op2")

        func = module.body.operations[0]
        entry_block = func.regions[0].blocks[0]
        ip = InsertionPoint.at_block_begin(entry_block)
        ip.insert(op1)
        ip.insert(op2)
        # CHECK: func @f1
        # CHECK: "custom.op1"()
        # CHECK: "custom.op2"()
        # CHECK: %0 = "custom.addi"
        print(module)

    # Trying to add a previously added op should raise.
    try:
        ip.insert(op1)
    except ValueError:
        pass
    else:
        assert False, "expected insert of attached op to raise"


# CHECK-LABEL: TEST: testOperationWithRegion
@run
def testOperationWithRegion():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        i32 = IntegerType.get_signed(32)
        op1 = Operation.create("custom.op1", regions=1)
        block = op1.regions[0].blocks.append(i32, i32)
        # CHECK: "custom.op1"() ({
        # CHECK: ^bb0(%arg0: si32, %arg1: si32):
        # CHECK:   "custom.terminator"() : () -> ()
        # CHECK: }) : () -> ()
        terminator = Operation.create("custom.terminator")
        ip = InsertionPoint(block)
        ip.insert(terminator)
        print(op1)

        # Now add the whole operation to another op.
        # TODO: Verify lifetime hazard by nulling out the new owning module and
        # accessing op1.
        # TODO: Also verify accessing the terminator once both parents are nulled
        # out.
        module = Module.parse(
            r"""
      func.func @f1(%arg0: i32) -> i32 {
        %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
        return %1 : i32
      }
    """
        )
        func = module.body.operations[0]
        entry_block = func.regions[0].blocks[0]
        ip = InsertionPoint.at_block_begin(entry_block)
        ip.insert(op1)
        # CHECK: func @f1
        # CHECK: "custom.op1"()
        # CHECK:   "custom.terminator"
        # CHECK: %0 = "custom.addi"
        print(module)


# CHECK-LABEL: TEST: testOperationResultList
@run
def testOperationResultList():
    ctx = Context()
    module = Module.parse(
        r"""
    func.func @f1() {
      %0:3 = call @f2() : () -> (i32, f64, index)
      call @f3() : () -> ()
      return
    }
    func.func private @f2() -> (i32, f64, index)
    func.func private @f3() -> ()
  """,
        ctx,
    )
    caller = module.body.operations[0]
    call = caller.regions[0].blocks[0].operations[0]
    assert len(call.results) == 3
    # CHECK: Result 0, type i32
    # CHECK: Result 1, type f64
    # CHECK: Result 2, type index
    for res in call.results:
        print(f"Result {res.result_number}, type {res.type}")

    # CHECK: Matched Result r0, type i32
    # CHECK: Matched Result r1, type f64
    # CHECK: Matched Result r2, type index
    match caller:
        case func.FuncOp(
            body=Region(
                blocks=[
                    Block(
                        operations=[OpView(results=[r0, r1, r2]) as matched_call, *_],
                    ),
                ],
            ),
        ):
            assert matched_call == call
            print(f"Matched Result r0, type {r0.type}")
            print(f"Matched Result r1, type {r1.type}")
            print(f"Matched Result r2, type {r2.type}")
        case _:
            assert False

    # CHECK: Result type i32
    # CHECK: Result type f64
    # CHECK: Result type index
    for t in call.results.types:
        print(f"Result type {t}")

    # Out of range
    expect_index_error(lambda: call.results[3])
    expect_index_error(lambda: call.results[-4])

    no_results_call = caller.regions[0].blocks[0].operations[1]
    assert len(no_results_call.results) == 0
    assert no_results_call.results.owner == no_results_call


# CHECK-LABEL: TEST: testOperationResultListSlice
@run
def testOperationResultListSlice():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(
            r"""
      func.func @f1() {
        "some.op"() : () -> (i1, i2, i3, i4, i5)
        return
      }
    """
        )
        func = module.body.operations[0]
        entry_block = func.regions[0].blocks[0]
        producer = entry_block.operations[0]

        assert len(producer.results) == 5
        for left, right in zip(producer.results, producer.results[::-1][::-1]):
            assert left == right
            assert left.result_number == right.result_number

        # CHECK: Result 0, type i1
        # CHECK: Result 1, type i2
        # CHECK: Result 2, type i3
        # CHECK: Result 3, type i4
        # CHECK: Result 4, type i5
        full_slice = producer.results[:]
        for res in full_slice:
            print(f"Result {res.result_number}, type {res.type}")

        # CHECK: Result 1, type i2
        # CHECK: Result 2, type i3
        # CHECK: Result 3, type i4
        middle = producer.results[1:4]
        for res in middle:
            print(f"Result {res.result_number}, type {res.type}")

        # CHECK: Result 1, type i2
        # CHECK: Result 3, type i4
        odd = producer.results[1::2]
        for res in odd:
            print(f"Result {res.result_number}, type {res.type}")

        # CHECK: Result 3, type i4
        # CHECK: Result 1, type i2
        inverted_middle = producer.results[-2:0:-2]
        for res in inverted_middle:
            print(f"Result {res.result_number}, type {res.type}")


# CHECK-LABEL: TEST: testOperationAttributes
@run
def testOperationAttributes():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    "some.op"() { some.attribute = 1 : i8,
                  other.attribute = 3.0,
                  dependent = "text" } : () -> ()
  """,
        ctx,
    )
    op = module.body.operations[0]
    assert len(op.attributes) == 3
    iattr = op.attributes["some.attribute"]
    fattr = op.attributes["other.attribute"]
    sattr = op.attributes["dependent"]
    # CHECK: Attribute type i8, value 1
    print(f"Attribute type {iattr.type}, value {iattr.value}")
    # CHECK: Attribute type f64, value 3.0
    print(f"Attribute type {fattr.type}, value {fattr.value}")
    # CHECK: Attribute value text
    print(f"Attribute value {sattr.value}")
    # CHECK: Attribute value b'text'
    print(f"Attribute value {sattr.value_bytes}")

    # Python dict-style iteration
    # We don't know in which order the attributes are stored.
    # CHECK-DAG: dependent
    # CHECK-DAG: other.attribute
    # CHECK-DAG: some.attribute
    for name in op.attributes:
        print(name)

    # Basic dict-like introspection
    # CHECK: True
    print("some.attribute" in op.attributes)
    # CHECK: False
    print("missing" in op.attributes)
    # CHECK: Keys: ['dependent', 'other.attribute', 'some.attribute']
    print("Keys:", sorted(op.attributes.keys()))
    # CHECK: Values count 3
    print("Values count", len(op.attributes.values()))
    # CHECK: Items count 3
    print("Items count", len(op.attributes.items()))

    # Dict() conversion test
    d = {k: v.value for k, v in dict(op.attributes).items()}
    # CHECK: Dict mapping {'dependent': 'text', 'other.attribute': 3.0, 'some.attribute': 1}
    print("Dict mapping", d)

    # Structural pattern matching test using Mapping

    # CHECK: Matched Mapping Attribute 'some.attribute': 1
    # CHECK: Matched Mapping Attribute 'other.attribute': 3.0
    # CHECK: Matched Mapping Attribute 'dependent': text
    match op:
        case OpView(attributes={"does_not_exist": a0}):
            assert False
        case OpView(
            attributes={
                "some.attribute": IntegerAttr(value=some_attr_val) as some_attr,
                "other.attribute": FloatAttr() as other_attr,
                "dependent": StringAttr() as dep_attr,
                **other_attributes,
            }
        ):
            print(f"Matched Mapping Attribute 'some.attribute': {some_attr_val}")
            print(f"Matched Mapping Attribute 'other.attribute': {other_attr.value}")
            print(f"Matched Mapping Attribute 'dependent': {dep_attr.value}")
            assert type(other_attributes) == dict
            assert len(other_attributes) == 0
            assert some_attr == op.attributes.get("some.attribute")
            assert other_attr == op.attributes.get("other.attribute", None)
            assert dep_attr == op.attributes.get("dependent", "Default value")
        case _:
            print("Did not match!")
            assert False

    # Check that exceptions are raised as expected.
    try:
        op.attributes["does_not_exist"]
    except KeyError:
        pass
    else:
        assert False, "expected KeyError on accessing a non-existent attribute"

    try:
        op.attributes[42]
    except IndexError:
        pass
    else:
        assert False, "expected IndexError on accessing an out-of-bounds attribute"

    # Check that exceptions are raised when `get` is used with non-str arg.
    try:
        op.attributes.get(0)
    except TypeError:
        pass
    else:
        assert False, "expected TypeError using int as key for get()"

    try:
        op.attributes.get(0, None)
    except TypeError:
        pass
    else:
        assert False, "expected TypeError using int as key for get()"

    try:
        op.attributes.get([], None)
    except TypeError:
        pass
    else:
        assert False, "expected TypeError using list as key for get()"

    try:
        match op:
            case OpView(attributes={0: a}):
                assert False
    except TypeError:
        pass
    else:
        assert False, "expected TypeError matching OpAttributeMap with int-key "

    # get() does not throw for non existent attributes.
    assert op.attributes.get("does_not_exist") is None
    assert op.attributes.get("does_not_exist", "default_value") == "default_value"


# CHECK-LABEL: TEST: testOperationPrint
@run
def testOperationPrint():
    ctx = Context()
    module = Module.parse(
        r"""
    func.func @f1(%arg0: i32) -> i32 {
      %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32> loc("nom")
      %1 = arith.constant dense_resource<resource1> : tensor<3xi64>
      return %arg0 : i32
    }

    {-#
      dialect_resources: {
          builtin: {
            resource1: "0x08000000010000000000000002000000000000000300000000000000"
          }
        }
      #-}
  """,
        ctx,
    )

    # Test print to stdout.
    # CHECK: return %arg0 : i32
    # CHECK: resource1: "0x08
    module.operation.print()

    # Test print to text file.
    f = io.StringIO()
    # CHECK: <class 'str'>
    # CHECK: return %arg0 : i32
    module.operation.print(file=f)
    str_value = f.getvalue()
    print(str_value.__class__)
    print(f.getvalue())

    # Test roundtrip to bytecode.
    bytecode_stream = io.BytesIO()
    module.operation.write_bytecode(bytecode_stream, desired_version=1)
    bytecode = bytecode_stream.getvalue()
    assert bytecode.startswith(b"ML\xefR"), "Expected bytecode to start with MLïR"
    with NamedTemporaryFile() as tmpfile:
        module.operation.write_bytecode(str(tmpfile.name), desired_version=1)
        tmpfile.seek(0)
        assert tmpfile.read().startswith(
            b"ML\xefR"
        ), "Expected bytecode to start with MLïR"
    ctx2 = Context()
    module_roundtrip = Module.parse(bytecode, ctx2)
    f = io.StringIO()
    module_roundtrip.operation.print(file=f)
    roundtrip_value = f.getvalue()
    assert str_value == roundtrip_value, "Mismatch after roundtrip bytecode"

    # Test print to binary file.
    f = io.BytesIO()
    # CHECK: <class 'bytes'>
    # CHECK: return %arg0 : i32
    module.operation.print(file=f, binary=True)
    bytes_value = f.getvalue()
    print(bytes_value.__class__)
    print(bytes_value)

    # Test print local_scope.
    # CHECK: constant dense<[1, 2, 3, 4]> : tensor<4xi32> loc("nom")
    module.operation.print(enable_debug_info=True, use_local_scope=True)
    # CHECK: %nom = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    module.operation.print(use_name_loc_as_prefix=True, use_local_scope=True)

    # Test printing using state.
    state = AsmState(module.operation)
    # CHECK: constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    module.operation.print(state)

    # Test print with options.
    # CHECK: value = dense_resource<__elided__> : tensor<4xi32>
    # CHECK: "func.return"(%arg0) : (i32) -> () -:5:7
    # CHECK-NOT: resource1: "0x08
    module.operation.print(
        large_elements_limit=2,
        enable_debug_info=True,
        pretty_debug_info=True,
        print_generic_op_form=True,
        use_local_scope=True,
    )

    # Test print with skip_regions option
    # CHECK: func.func @f1(%arg0: i32) -> i32
    # CHECK-NOT: func.return
    module.body.operations[0].print(
        skip_regions=True,
    )

    # Test print with large_resource_limit.
    # CHECK: func.func @f1(%arg0: i32) -> i32
    # CHECK-NOT: resource1: "0x08
    module.operation.print(large_resource_limit=2)

    # Test large_elements_limit has no effect on resource string
    # CHECK: func.func @f1(%arg0: i32) -> i32
    # CHECK: resource1: "0x08
    module.operation.print(large_elements_limit=2)


# CHECK-LABEL: TEST: testKnownOpView
@run
def testKnownOpView():
    with Context(), Location.unknown():
        Context.current.allow_unregistered_dialects = True
        module = Module.parse(
            r"""
      %1 = "custom.f32"() : () -> f32
      %2 = "custom.f32"() : () -> f32
      %3 = arith.addf %1, %2 : f32
      %4 = arith.constant 0 : i32
    """
        )
        print(module)

        # addf should map to a known OpView class in the arithmetic dialect.
        # We know the OpView for it defines an 'lhs' attribute.
        addf = module.body.operations[2]
        # CHECK: <mlir.dialects._arith_ops_gen.AddFOp object
        print(repr(addf))
        # CHECK: "custom.f32"()
        print(addf.lhs)

        # One of the custom ops should resolve to the default OpView.
        custom = module.body.operations[0]
        # CHECK: OpView object
        print(repr(custom))

        # Check again to make sure negative caching works.
        custom = module.body.operations[0]
        # CHECK: OpView object
        print(repr(custom))

        # constant should map to an extension OpView class in the arithmetic dialect.
        constant = module.body.operations[3]
        # CHECK: <mlir.dialects.arith.ConstantOp object
        print(repr(constant))
        # Checks that the arith extension is being registered successfully
        # (literal_value is a property on the extension class but not on the default OpView).
        # CHECK: literal value 0
        print("literal value", constant.literal_value)

        # Checks that "late" registration/replacement (i.e., post all module loading/initialization)
        # is working correctly.
        @_cext.register_operation(arith._Dialect, replace=True)
        class ConstantOp(arith.ConstantOp):
            def __init__(self, result, value, *, loc=None, ip=None):
                if isinstance(value, int):
                    super().__init__(IntegerAttr.get(result, value), loc=loc, ip=ip)
                elif isinstance(value, float):
                    super().__init__(FloatAttr.get(result, value), loc=loc, ip=ip)
                else:
                    super().__init__(value, loc=loc, ip=ip)

        constant = module.body.operations[3]
        # CHECK: <__main__.testKnownOpView.<locals>.ConstantOp object
        print(repr(constant))


# CHECK-LABEL: TEST: testFailedGenericOperationCreationReportsError
@run
def testFailedGenericOperationCreationReportsError():
    with Context(), Location.unknown():
        c0 = shape.const_shape([])
        c1 = shape.const_shape([1, 2, 3])
        try:
            shape.MeetOp.build_generic(operands=[c0, c1])
        except MLIRError as e:
            # CHECK: unequal shape cardinality
            print(e)
        else:
            assert False, "Expected exception"


# CHECK-LABEL: TEST: testSingleResultProperty
@run
def testSingleResultProperty():
    with Context(), Location.unknown():
        Context.current.allow_unregistered_dialects = True
        module = Module.parse(
            r"""
      "custom.no_result"() : () -> ()
      %0:2 = "custom.two_result"() : () -> (f32, f32)
      %1 = "custom.one_result"() : () -> f32
    """
        )
        print(module)

    try:
        module.body.operations[0].result
    except ValueError as e:
        # CHECK: Cannot call .result on operation custom.no_result which has 0 results
        print(e)
    else:
        assert False, "Expected exception"

    try:
        module.body.operations[1].result
    except ValueError as e:
        # CHECK: Cannot call .result on operation custom.two_result which has 2 results
        print(e)
    else:
        assert False, "Expected exception"

    # CHECK: %1 = "custom.one_result"() : () -> f32
    print(module.body.operations[2])


def create_invalid_operation():
    # This module has two region and is invalid verify that we fallback
    # to the generic printer for safety.
    op = Operation.create("builtin.module", regions=2)
    op.regions[0].blocks.append()
    return op


# CHECK-LABEL: TEST: testInvalidOperationStrSoftFails
@run
def testInvalidOperationStrSoftFails():
    ctx = Context()
    with Location.unknown(ctx):
        invalid_op = create_invalid_operation()
        # Verify that we fallback to the generic printer for safety.
        # CHECK: "builtin.module"() ({
        # CHECK: }) : () -> ()
        print(invalid_op)
        try:
            invalid_op.verify()
        except MLIRError as e:
            # CHECK: Exception: <
            # CHECK:   Verification failed:
            # CHECK:   error: unknown: 'builtin.module' op requires one region
            # CHECK:    note: unknown: see current operation:
            # CHECK:     "builtin.module"() ({
            # CHECK:     ^bb0:
            # CHECK:     }, {
            # CHECK:     }) : () -> ()
            # CHECK: >
            print(f"Exception: <{e}>")


# CHECK-LABEL: TEST: testInvalidModuleStrSoftFails
@run
def testInvalidModuleStrSoftFails():
    ctx = Context()
    with Location.unknown(ctx):
        module = Module.create()
        with InsertionPoint(module.body):
            invalid_op = create_invalid_operation()
        # Verify that we fallback to the generic printer for safety.
        # CHECK: "builtin.module"() ({
        # CHECK: }) : () -> ()
        print(module)


# CHECK-LABEL: TEST: testInvalidOperationGetAsmBinarySoftFails
@run
def testInvalidOperationGetAsmBinarySoftFails():
    ctx = Context()
    with Location.unknown(ctx):
        invalid_op = create_invalid_operation()
        # Verify that we fallback to the generic printer for safety.
        # CHECK: b'"builtin.module"() ({\n^bb0:\n}, {\n}) : () -> ()\n'
        print(invalid_op.get_asm(binary=True))


# CHECK-LABEL: TEST: testCreateWithInvalidAttributes
@run
def testCreateWithInvalidAttributes():
    ctx = Context()
    with Location.unknown(ctx):
        try:
            Operation.create(
                "builtin.module", attributes={None: StringAttr.get("name")}
            )
        except Exception as e:
            # CHECK: Invalid attribute key (not a string) when attempting to create the operation "builtin.module"
            print(e)
        try:
            Operation.create("builtin.module", attributes={42: StringAttr.get("name")})
        except Exception as e:
            # CHECK: Invalid attribute key (not a string) when attempting to create the operation "builtin.module"
            print(e)
        try:
            Operation.create("builtin.module", attributes={"some_key": ctx})
        except Exception as e:
            # CHECK: Invalid attribute value for the key "some_key" when attempting to create the operation "builtin.module"
            print(e)
        try:
            Operation.create("builtin.module", attributes={"some_key": None})
        except Exception as e:
            # CHECK: Found an invalid (`None`?) attribute value for the key "some_key" when attempting to create the operation "builtin.module"
            print(e)


# CHECK-LABEL: TEST: testOperationName
@run
def testOperationName():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    %0 = "custom.op1"() : () -> f32
    %1 = "custom.op2"() : () -> i32
    %2 = "custom.op1"() : () -> f32
  """,
        ctx,
    )

    # CHECK: custom.op1
    # CHECK: custom.op2
    # CHECK: custom.op1
    for op in module.body.operations:
        print(op.operation.name)


# CHECK-LABEL: TEST: testCapsuleConversions
@run
def testCapsuleConversions():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        m = Operation.create("custom.op1").operation
        m_capsule = m._CAPIPtr
        assert '"mlir.ir.Operation._CAPIPtr"' in repr(m_capsule)
        m2 = Operation._CAPICreate(m_capsule)
        assert m2 is not m
        assert m2 == m
        # Gc and verify destructed.
        m = None
        m_capsule = None
        m2 = None
        gc.collect()


# CHECK-LABEL: TEST: testOperationErase
@run
def testOperationErase():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        m = Module.create()
        with InsertionPoint(m.body):
            op = Operation.create("custom.op1")

            # CHECK: "custom.op1"
            print(m)

            op.operation.erase()

            # CHECK-NOT: "custom.op1"
            print(m)

            # Ensure we can create another operation
            Operation.create("custom.op2")


# CHECK-LABEL: TEST: testOperationClone
@run
def testOperationClone():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        m = Module.create()
        with InsertionPoint(m.body):
            op = Operation.create("custom.op1")

            # CHECK: "custom.op1"
            print(m)

            clone = op.operation.clone()
            op.operation.erase()

            # CHECK: "custom.op1"
            print(m)


# CHECK-LABEL: TEST: testOperationLoc
@run
def testOperationLoc():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        loc = Location.name("loc")
        op = Operation.create("custom.op", loc=loc)
        assert op.location == loc
        assert op.operation.location == loc

        another_loc = Location.name("another_loc")
        op.location = another_loc
        assert op.location == another_loc
        assert op.operation.location == another_loc
        # CHECK: loc("another_loc")
        print(op.location)


# CHECK-LABEL: TEST: testModuleMerge
@run
def testModuleMerge():
    with Context():
        m1 = Module.parse("func.func private @foo()")
        m2 = Module.parse(
            """
      func.func private @bar()
      func.func private @qux()
    """
        )
        foo = m1.body.operations[0]
        bar = m2.body.operations[0]
        qux = m2.body.operations[1]
        assert bar.is_before_in_block(qux)
        bar.move_before(foo)
        assert bar.is_before_in_block(foo)
        qux.move_after(foo)
        assert bar.is_before_in_block(qux)
        assert foo.is_before_in_block(qux)

        # CHECK: module
        # CHECK: func private @bar
        # CHECK: func private @foo
        # CHECK: func private @qux
        print(m1)

        # CHECK: module {
        # CHECK-NEXT: }
        print(m2)


# CHECK-LABEL: TEST: testAppendMoveFromAnotherBlock
@run
def testAppendMoveFromAnotherBlock():
    with Context():
        m1 = Module.parse("func.func private @foo()")
        m2 = Module.parse("func.func private @bar()")
        func = m1.body.operations[0]
        m2.body.append(func)

        # CHECK: module
        # CHECK: func private @bar
        # CHECK: func private @foo

        print(m2)
        # CHECK: module {
        # CHECK-NEXT: }
        print(m1)


# CHECK-LABEL: TEST: testDetachFromParent
@run
def testDetachFromParent():
    with Context():
        m1 = Module.parse("func.func private @foo()")
        func = m1.body.operations[0].detach_from_parent()
        # CHECK: func.attached=False
        print(f"{func.attached=}")

        try:
            func.detach_from_parent()
        except ValueError as e:
            if "has no parent" not in str(e):
                raise
        else:
            assert False, "expected ValueError when detaching a detached operation"

        print(m1)
        # CHECK-NOT: func private @foo


# CHECK-LABEL: TEST: testOperationHash
@run
def testOperationHash():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        op = Operation.create("custom.op1")
        assert hash(op) == hash(op.operation)

        module = Module.create()
        with InsertionPoint(module.body):
            op2 = Operation.create("custom.op2")
            custom_op2 = module.body.operations[0]
            assert hash(op2) == hash(custom_op2)


# CHECK-LABEL: TEST: testOperationParse
@run
def testOperationParse():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True

        # Generic operation parsing.
        m = Operation.parse("module {}")
        o = Operation.parse('"test.foo"() : () -> ()')
        assert isinstance(m, ModuleOp)
        assert type(o) is OpView

        # Parsing specific operation.
        m = ModuleOp.parse("module {}")
        assert isinstance(m, ModuleOp)
        try:
            ModuleOp.parse('"test.foo"() : () -> ()')
        except MLIRError as e:
            # CHECK: error: Expected a 'builtin.module' op, got: 'test.foo'
            print(f"error: {e}")
        else:
            assert False, "expected error"

        o = Operation.parse('"test.foo"() : () -> ()', source_name="my-source-string")
        # CHECK: op_with_source_name: "test.foo"() : () -> () loc("my-source-string":1:1)
        print(
            f"op_with_source_name: {o.get_asm(enable_debug_info=True, use_local_scope=True)}"
        )


# CHECK-LABEL: TEST: testOpWalk
@run
def testOpWalk():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        r"""
    builtin.module {
      func.func @f() {
        func.return
      }
    }
  """,
        ctx,
    )

    def callback(op):
        print(op.name)
        return WalkResult.ADVANCE

    # Test post-order walk (default).
    # CHECK-NEXT:  Post-order
    # CHECK-NEXT:  func.return
    # CHECK-NEXT:  func.func
    # CHECK-NEXT:  builtin.module
    print("Post-order")
    module.operation.walk(callback)

    # Test pre-order walk.
    # CHECK-NEXT:  Pre-order
    # CHECK-NEXT:  builtin.module
    # CHECK-NEXT:  func.fun
    # CHECK-NEXT:  func.return
    print("Pre-order")
    module.operation.walk(callback, WalkOrder.PRE_ORDER)

    # Test interrput.
    # CHECK-NEXT:  Interrupt post-order
    # CHECK-NEXT:  func.return
    print("Interrupt post-order")

    def callback(op):
        print(op.name)
        return WalkResult.INTERRUPT

    module.operation.walk(callback)

    # Test skip.
    # CHECK-NEXT:  Skip pre-order
    # CHECK-NEXT:  builtin.module
    print("Skip pre-order")

    def callback(op):
        print(op.name)
        return WalkResult.SKIP

    module.operation.walk(callback, WalkOrder.PRE_ORDER)

    # Test exception.
    # CHECK: Exception
    # CHECK-NEXT: func.return
    # CHECK-NEXT: Exception raised
    print("Exception")

    def callback(op):
        print(op.name)
        raise ValueError
        return WalkResult.ADVANCE

    try:
        module.operation.walk(callback)
    except RuntimeError:
        print("Exception raised")

    # Test op_class filter: only visits ops of the requested type.
    module = Module.parse(
        r"""
    module {
      func.func @f() {
        func.return
      }
      func.func @g() {
        func.return
      }
      arith.constant dense<0> : tensor<i32>
    }
  """,
        ctx,
    )

    # CHECK-NEXT: only FuncOp visited: True
    only_funcs = True

    def check_type(op):
        nonlocal only_funcs
        if not isinstance(op.opview, func.FuncOp):
            only_funcs = False
        return WalkResult.ADVANCE

    module.operation.walk(check_type, op_class=func.FuncOp)
    print(f"only FuncOp visited: {only_funcs}")

    # CHECK-NEXT: interrupted after: 1
    seen = []

    def stop_after_first(op):
        seen.append(op.opview)
        return WalkResult.INTERRUPT

    module.operation.walk(stop_after_first, op_class=func.FuncOp)
    print(f"interrupted after: {len(seen)}")

    # CHECK-NEXT: never called: True
    called = False

    def should_not_run(op):
        nonlocal called
        called = True
        return WalkResult.ADVANCE

    module.operation.walk(should_not_run, op_class=scf.ForOp)
    print(f"never called: {not called}")

    # CHECK-NEXT: collected func.FuncOp: ['"f"', '"g"']
    collected = []

    def collect(op):
        collected.append(op.opview)
        return WalkResult.ADVANCE

    module.operation.walk(collect, op_class=func.FuncOp)
    assert all(isinstance(r, func.FuncOp) for r in collected)
    print(f"collected func.FuncOp: {[str(r.name) for r in collected]}")

    # Test op_class with walk_order: pre-order visits FuncOps in source order.
    # CHECK-NEXT: pre-order FuncOp names: ['"f"', '"g"']
    collected.clear()
    module.operation.walk(collect, WalkOrder.PRE_ORDER, op_class=func.FuncOp)
    assert all(isinstance(r, func.FuncOp) for r in collected)
    print(f"pre-order FuncOp names: {[str(r.name) for r in collected]}")


# CHECK-LABEL: TEST: testOpReplaceUsesWith
@run
def testOpReplaceUsesWith():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with Location.unknown(ctx):
        m = Module.create()
        i32 = IntegerType.get_signless(32)
        with InsertionPoint(m.body):
            value = Operation.create("custom.op1", results=[i32]).results[0]
            value2 = Operation.create("custom.op2", results=[i32]).results[0]
            op = Operation.create("custom.op3", operands=[value])
            op2 = Operation.create("custom.op4", operands=[value])
            op.replace_uses_of_with(value, value2)

    assert len(list(value.uses)) == 1

    # CHECK: Use owner: "custom.op4"
    # CHECK: Use operand_number: 0
    for use in value.uses:
        assert use.owner in [op2]
        print(f"Use owner: {use.owner}")
        print(f"Use operand_number: {use.operand_number}")

    assert len(list(value2.uses)) == 1

    # CHECK: Use owner: "custom.op3"
    # CHECK: Use operand_number: 0
    for use in value2.uses:
        assert use.owner in [op]
        print(f"Use owner: {use.owner}")
        print(f"Use operand_number: {use.operand_number}")


# CHECK-LABEL: TEST: testGetOwnerConcreteOpview
@run
def testGetOwnerConcreteOpview():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = arith.ConstantOp(value=42, result=IntegerType.get_signless(32))
            r = arith.AddIOp(a, a, overflowFlags=arith.IntegerOverflowFlags.nsw)
            for u in a.result.uses:
                assert isinstance(u.owner, arith.AddIOp)


# CHECK-LABEL: TEST: testIndexSwitch
@run
def testIndexSwitch():
    with Context() as ctx, Location.unknown():
        i32 = T.i32()
        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(T.index())
            def index_switch(index):
                c1 = arith.constant(i32, 1)
                switch_op = scf.IndexSwitchOp(results=[i32], arg=index, cases=range(3))

                assert len(switch_op.regions) == 4
                assert len(switch_op.regions[2:]) == 2
                assert len([i for i in switch_op.regions[2:]]) == 2
                assert len(switch_op.caseRegions) == 3
                assert len([i for i in switch_op.caseRegions]) == 3
                assert len(switch_op.caseRegions[1:]) == 2
                assert len([i for i in switch_op.caseRegions[1:]]) == 2


# CHECK-LABEL: TEST: testGetParentOfType
@run
def testGetParentOfType():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        idx = IndexType.get()
        # Build: func.func -> scf.for -> custom.base_op
        func_op: func.FuncOp = func.FuncOp("test_fn", ([], []))
        with InsertionPoint(func_op.add_entry_block()):
            lower_bound = arith.ConstantOp(idx, 0)
            upper_bound = arith.ConstantOp(idx, 10)
            step = arith.ConstantOp(idx, 1)
            for_op: scf.ForOp = scf.ForOp(lower_bound, upper_bound, step)
            with InsertionPoint(for_op.body):
                base_op: Operation = Operation.create("custom.base_op")
                scf.YieldOp([])
            func.ReturnOp([])

        # CHECK: get_parent_of_type detached->func.func: None
        detached: Operation = Operation.create("custom.detached")
        res = get_parent_of_type(detached, func.FuncOp)
        print(f"get_parent_of_type detached->func.func: {res}")
        assert res is None

        # CHECK: get_parent_of_type base_op->func.func: func.func
        res = get_parent_of_type(base_op, func.FuncOp)
        print(f"get_parent_of_type base_op->func.func: {res.operation.name}")
        assert isinstance(res, func.FuncOp)

        # CHECK: get_parent_of_type func_op->func.func: None
        res = get_parent_of_type(func_op, func.FuncOp)
        print(f"get_parent_of_type func_op->func.func: {res}")
        assert res is None

        # CHECK: get_parent_of_type base_op->scf.if: None
        res = get_parent_of_type(base_op, scf.IfOp)
        print(f"get_parent_of_type base_op->scf.if: {res}")
        assert res is None

        try:
            get_parent_of_type(base_op, int)
            assert False, "expected TypeError"
        except TypeError:
            pass


# CHECK-LABEL: TEST: test_get_ops_of_type
@run
def test_get_ops_of_type():
    with Context(), Location.unknown():
        module = Module.parse(
            r"""
    module {
      func.func @f() {
        func.return
      }
      func.func @g() {
        func.return
      }
    }
  """
        )

        # CHECK: get_ops_of_type func.func count: 2
        results = get_ops_of_type(module, func.FuncOp)
        print(f"get_ops_of_type func.func count: {len(results)}")
        assert len(results) == 2
        assert all(isinstance(r, func.FuncOp) for r in results)

        # CHECK: get_ops_of_type scf.for count: 0
        results = get_ops_of_type(module, scf.ForOp)
        print(f"get_ops_of_type scf.for count: {len(results)}")
        assert len(results) == 0

        # CHECK: get_ops_of_type func_op->func.ReturnOp count: 1
        # Accepts OpView as root.
        func_op = get_ops_of_type(module, func.FuncOp)[0]
        results = get_ops_of_type(func_op, func.ReturnOp)
        print(f"get_ops_of_type func_op->func.ReturnOp count: {len(results)}")
        assert len(results) == 1
        assert isinstance(results[0], func.ReturnOp)

        # CHECK: get_ops_of_type no filter count: 5
        # No op_class collects all ops.
        results = get_ops_of_type(module)
        print(f"get_ops_of_type no filter count: {len(results)}")
        assert len(results) == 5
        assert any(isinstance(r, func.FuncOp) for r in results)
        assert any(isinstance(r, func.ReturnOp) for r in results)
