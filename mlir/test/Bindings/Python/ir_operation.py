# RUN: %PYTHON %s | FileCheck %s

import gc
import itertools
import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0


# Verify iterator based traversal of the op/region/block hierarchy.
# CHECK-LABEL: TEST: testTraverseOpRegionBlockIterators
def testTraverseOpRegionBlockIterators():
  ctx = mlir.ir.Context()
  ctx.allow_unregistered_dialects = True
  module = ctx.parse_module(r"""
    func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """)
  op = module.operation
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
  ctx = mlir.ir.Context()
  ctx.allow_unregistered_dialects = True
  module = ctx.parse_module(r"""
    func @f1(%arg0: i32) -> i32 {
      %1 = "custom.addi"(%arg0, %arg0) : (i32, i32) -> i32
      return %1 : i32
    }
  """)

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
