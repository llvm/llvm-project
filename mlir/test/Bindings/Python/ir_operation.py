# RUN: %PYTHON %s | FileCheck %s

import gc
import io
import itertools
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# Verify iterator based traversal of the op/region/block hierarchy.
# CHECK-LABEL: TEST: testTraverseOpRegionBlockIterators
def testTraverseOpRegionBlockIterators():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(r"""
    func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """, ctx)
  op = module.operation
  assert op.context is ctx
  # Get the block using iterators off of the named collections.
  regions = list(op.regions)
  blocks = list(regions[0].blocks)
  # CHECK: MODULE REGIONS=1 BLOCKS=1
  print(f"MODULE REGIONS={len(regions)} BLOCKS={len(blocks)}")

  # Should verify.
  # CHECK: .verify = True
  print(f".verify = {module.operation.verify()}")

  # Get the regions and blocks from the default collections.
  default_regions = list(op)
  default_blocks = list(default_regions[0])
  # They should compare equal regardless of how obtained.
  assert default_regions == regions
  assert default_blocks == blocks

  # Should be able to get the operations from either the named collection
  # or the block.
  operations = list(blocks[0].operations)
  default_operations = list(blocks[0])
  assert default_operations == operations

  def walk_operations(indent, op):
    for i, region in enumerate(op):
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
  # CHECK:           OP 1: return
  # CHECK:    OP 1: module_terminator
  walk_operations("", op)

run(testTraverseOpRegionBlockIterators)


# Verify index based traversal of the op/region/block hierarchy.
# CHECK-LABEL: TEST: testTraverseOpRegionBlockIndices
def testTraverseOpRegionBlockIndices():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(r"""
    func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """, ctx)

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
          walk_operations(indent + "      ", child_op)

  # CHECK: REGION 0:
  # CHECK:   BLOCK 0:
  # CHECK:     OP 0: func
  # CHECK:       REGION 0:
  # CHECK:         BLOCK 0:
  # CHECK:           OP 0: %0 = "custom.addi"
  # CHECK:           OP 1: return
  # CHECK:    OP 1: module_terminator
  walk_operations("", module.operation)

run(testTraverseOpRegionBlockIndices)


# CHECK-LABEL: TEST: testBlockArgumentList
def testBlockArgumentList():
  with Context() as ctx:
    module = Module.parse(r"""
      func @f1(%arg0: i32, %arg1: f64, %arg2: index) {
        return
      }
    """, ctx)
    func = module.body.operations[0]
    entry_block = func.regions[0].blocks[0]
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


run(testBlockArgumentList)


# CHECK-LABEL: TEST: testOperationOperands
def testOperationOperands():
  with Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = Module.parse(r"""
      func @f1(%arg0: i32) {
        %0 = "test.producer"() : () -> i64
        "test.consumer"(%arg0, %0) : (i32, i64) -> ()
        return
      }""")
    func = module.body.operations[0]
    entry_block = func.regions[0].blocks[0]
    consumer = entry_block.operations[1]
    assert len(consumer.operands) == 2
    # CHECK: Operand 0, type i32
    # CHECK: Operand 1, type i64
    for i, operand in enumerate(consumer.operands):
      print(f"Operand {i}, type {operand.type}")


run(testOperationOperands)


# CHECK-LABEL: TEST: testOperationOperandsSlice
def testOperationOperandsSlice():
  with Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = Module.parse(r"""
      func @f1() {
        %0 = "test.producer0"() : () -> i64
        %1 = "test.producer1"() : () -> i64
        %2 = "test.producer2"() : () -> i64
        %3 = "test.producer3"() : () -> i64
        %4 = "test.producer4"() : () -> i64
        "test.consumer"(%0, %1, %2, %3, %4) : (i64, i64, i64, i64, i64) -> ()
        return
      }""")
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


run(testOperationOperandsSlice)


# CHECK-LABEL: TEST: testDetachedOperation
def testDetachedOperation():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    i32 = IntegerType.get_signed(32)
    op1 = Operation.create(
        "custom.op1", results=[i32, i32], regions=1, attributes={
            "foo": StringAttr.get("foo_value"),
            "bar": StringAttr.get("bar_value"),
        })
    # CHECK: %0:2 = "custom.op1"() ( {
    # CHECK: }) {bar = "bar_value", foo = "foo_value"} : () -> (si32, si32)
    print(op1)

  # TODO: Check successors once enough infra exists to do it properly.

run(testDetachedOperation)


# CHECK-LABEL: TEST: testOperationInsertionPoint
def testOperationInsertionPoint():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(r"""
    func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """, ctx)

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

run(testOperationInsertionPoint)


# CHECK-LABEL: TEST: testOperationWithRegion
def testOperationWithRegion():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    i32 = IntegerType.get_signed(32)
    op1 = Operation.create("custom.op1", regions=1)
    block = op1.regions[0].blocks.append(i32, i32)
    # CHECK: "custom.op1"() ( {
    # CHECK: ^bb0(%arg0: si32, %arg1: si32):  // no predecessors
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
    module = Module.parse(r"""
      func @f1(%arg0: i32) -> i32 {
        %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
        return %1 : i32
      }
    """)
    func = module.body.operations[0]
    entry_block = func.regions[0].blocks[0]
    ip = InsertionPoint.at_block_begin(entry_block)
    ip.insert(op1)
    # CHECK: func @f1
    # CHECK: "custom.op1"()
    # CHECK:   "custom.terminator"
    # CHECK: %0 = "custom.addi"
    print(module)

run(testOperationWithRegion)


# CHECK-LABEL: TEST: testOperationResultList
def testOperationResultList():
  ctx = Context()
  module = Module.parse(r"""
    func @f1() {
      %0:3 = call @f2() : () -> (i32, f64, index)
      return
    }
    func private @f2() -> (i32, f64, index)
  """, ctx)
  caller = module.body.operations[0]
  call = caller.regions[0].blocks[0].operations[0]
  assert len(call.results) == 3
  # CHECK: Result 0, type i32
  # CHECK: Result 1, type f64
  # CHECK: Result 2, type index
  for res in call.results:
    print(f"Result {res.result_number}, type {res.type}")


run(testOperationResultList)


# CHECK-LABEL: TEST: testOperationResultListSlice
def testOperationResultListSlice():
  with Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = Module.parse(r"""
      func @f1() {
        "some.op"() : () -> (i1, i2, i3, i4, i5)
        return
      }
    """)
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


run(testOperationResultListSlice)


# CHECK-LABEL: TEST: testOperationAttributes
def testOperationAttributes():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(r"""
    "some.op"() { some.attribute = 1 : i8,
                  other.attribute = 3.0,
                  dependent = "text" } : () -> ()
  """, ctx)
  op = module.body.operations[0]
  assert len(op.attributes) == 3
  iattr = IntegerAttr(op.attributes["some.attribute"])
  fattr = FloatAttr(op.attributes["other.attribute"])
  sattr = StringAttr(op.attributes["dependent"])
  # CHECK: Attribute type i8, value 1
  print(f"Attribute type {iattr.type}, value {iattr.value}")
  # CHECK: Attribute type f64, value 3.0
  print(f"Attribute type {fattr.type}, value {fattr.value}")
  # CHECK: Attribute value text
  print(f"Attribute value {sattr.value}")

  # We don't know in which order the attributes are stored.
  # CHECK-DAG: NamedAttribute(dependent="text")
  # CHECK-DAG: NamedAttribute(other.attribute=3.000000e+00 : f64)
  # CHECK-DAG: NamedAttribute(some.attribute=1 : i8)
  for attr in op.attributes:
    print(str(attr))

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


run(testOperationAttributes)


# CHECK-LABEL: TEST: testOperationPrint
def testOperationPrint():
  ctx = Context()
  module = Module.parse(r"""
    func @f1(%arg0: i32) -> i32 {
      %0 = constant dense<[1, 2, 3, 4]> : tensor<4xi32>
      return %arg0 : i32
    }
  """, ctx)

  # Test print to stdout.
  # CHECK: return %arg0 : i32
  module.operation.print()

  # Test print to text file.
  f = io.StringIO()
  # CHECK: <class 'str'>
  # CHECK: return %arg0 : i32
  module.operation.print(file=f)
  str_value = f.getvalue()
  print(str_value.__class__)
  print(f.getvalue())

  # Test print to binary file.
  f = io.BytesIO()
  # CHECK: <class 'bytes'>
  # CHECK: return %arg0 : i32
  module.operation.print(file=f, binary=True)
  bytes_value = f.getvalue()
  print(bytes_value.__class__)
  print(bytes_value)

  # Test get_asm with options.
  # CHECK: value = opaque<"", "0xDEADBEEF"> : tensor<4xi32>
  # CHECK: "std.return"(%arg0) : (i32) -> () -:4:7
  module.operation.print(large_elements_limit=2, enable_debug_info=True,
      pretty_debug_info=True, print_generic_op_form=True, use_local_scope=True)

run(testOperationPrint)


# CHECK-LABEL: TEST: testKnownOpView
def testKnownOpView():
  with Context(), Location.unknown():
    Context.current.allow_unregistered_dialects = True
    module = Module.parse(r"""
      %1 = "custom.f32"() : () -> f32
      %2 = "custom.f32"() : () -> f32
      %3 = addf %1, %2 : f32
    """)
    print(module)

    # addf should map to a known OpView class in the std dialect.
    # We know the OpView for it defines an 'lhs' attribute.
    addf = module.body.operations[2]
    # CHECK: <mlir.dialects._std_ops_gen._AddFOp object
    print(repr(addf))
    # CHECK: "custom.f32"()
    print(addf.lhs)

    # One of the custom ops should resolve to the default OpView.
    custom = module.body.operations[0]
    # CHECK: <_mlir.ir.OpView object
    print(repr(custom))

    # Check again to make sure negative caching works.
    custom = module.body.operations[0]
    # CHECK: <_mlir.ir.OpView object
    print(repr(custom))

run(testKnownOpView)


# CHECK-LABEL: TEST: testSingleResultProperty
def testSingleResultProperty():
  with Context(), Location.unknown():
    Context.current.allow_unregistered_dialects = True
    module = Module.parse(r"""
      "custom.no_result"() : () -> ()
      %0:2 = "custom.two_result"() : () -> (f32, f32)
      %1 = "custom.one_result"() : () -> f32
    """)
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

run(testSingleResultProperty)

# CHECK-LABEL: TEST: testPrintInvalidOperation
def testPrintInvalidOperation():
  ctx = Context()
  with Location.unknown(ctx):
    module = Operation.create("module", regions=1)
    # This block does not have a terminator, it may crash the custom printer.
    # Verify that we fallback to the generic printer for safety.
    block = module.regions[0].blocks.append()
    # CHECK: // Verification failed, printing generic form
    # CHECK: "module"() ( {
    # CHECK: }) : () -> ()
    print(module)
    # CHECK: .verify = False
    print(f".verify = {module.operation.verify()}")
run(testPrintInvalidOperation)


# CHECK-LABEL: TEST: testCreateWithInvalidAttributes
def testCreateWithInvalidAttributes():
  ctx = Context()
  with Location.unknown(ctx):
    try:
      Operation.create("module", attributes={None:StringAttr.get("name")})
    except Exception as e:
      # CHECK: Invalid attribute key (not a string) when attempting to create the operation "module"
      print(e)
    try:
      Operation.create("module", attributes={42:StringAttr.get("name")})
    except Exception as e:
      # CHECK: Invalid attribute key (not a string) when attempting to create the operation "module"
      print(e)
    try:
      Operation.create("module", attributes={"some_key":ctx})
    except Exception as e:
      # CHECK: Invalid attribute value for the key "some_key" when attempting to create the operation "module"
      print(e)
    try:
      Operation.create("module", attributes={"some_key":None})
    except Exception as e:
      # CHECK: Found an invalid (`None`?) attribute value for the key "some_key" when attempting to create the operation "module"
      print(e)
run(testCreateWithInvalidAttributes)


# CHECK-LABEL: TEST: testOperationName
def testOperationName():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(r"""
    %0 = "custom.op1"() : () -> f32
    %1 = "custom.op2"() : () -> i32
    %2 = "custom.op1"() : () -> f32
  """, ctx)

  # CHECK: custom.op1
  # CHECK: custom.op2
  # CHECK: custom.op1
  for op in module.body.operations:
    print(op.operation.name)

run(testOperationName)
