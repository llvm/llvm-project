# RUN: %PYTHON %s | FileCheck %s

import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()


# CHECK-LABEL: TEST: testDetachedRegionBlock
def testDetachedRegionBlock():
  ctx = mlir.ir.Context()
  t = mlir.ir.F32Type(ctx)
  region = ctx.create_region()
  block = ctx.create_block([t, t])
  # CHECK: <<UNLINKED BLOCK>>
  print(block)

run(testDetachedRegionBlock)


# CHECK-LABEL: TEST: testBlockTypeContextMismatch
def testBlockTypeContextMismatch():
  c1 = mlir.ir.Context()
  c2 = mlir.ir.Context()
  t1 = mlir.ir.F32Type(c1)
  t2 = mlir.ir.F32Type(c2)
  try:
    block = c1.create_block([t1, t2])
  except ValueError as e:
    # CHECK: ERROR: All types used to construct a block must be from the same context as the block
    print("ERROR:", e)

run(testBlockTypeContextMismatch)


# CHECK-LABEL: TEST: testBlockAppend
def testBlockAppend():
  ctx = mlir.ir.Context()
  t = mlir.ir.F32Type(ctx)
  region = ctx.create_region()
  try:
    region.first_block
  except IndexError:
    pass
  else:
    raise RuntimeError("Expected exception not raised")

  block = ctx.create_block([t, t])
  region.append_block(block)
  try:
    region.append_block(block)
  except ValueError:
    pass
  else:
    raise RuntimeError("Expected exception not raised")

  block2 = ctx.create_block([t])
  region.insert_block(1, block2)
  # CHECK: <<UNLINKED BLOCK>>
  block_first = region.first_block
  print(block_first)
  block_next = block_first.next_in_region
  try:
    block_next = block_next.next_in_region
  except IndexError:
    pass
  else:
    raise RuntimeError("Expected exception not raised")

run(testBlockAppend)
