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
  # CHECK:    OP 1: "module_terminator"
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
  # CHECK:    OP 1: "module_terminator"
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
    func @f2() -> (i32, f64, index)
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
