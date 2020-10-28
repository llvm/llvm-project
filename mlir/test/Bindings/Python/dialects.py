# RUN: %PYTHON %s | FileCheck %s

import gc
import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0


# CHECK-LABEL: TEST: testDialectDescriptor
def testDialectDescriptor():
  ctx = mlir.ir.Context()
  d = ctx.get_dialect_descriptor("std")
  # CHECK: <DialectDescriptor std>
  print(d)
  # CHECK: std
  print(d.namespace)
  try:
    _ = ctx.get_dialect_descriptor("not_existing")
  except ValueError:
    pass
  else:
    assert False, "Expected exception"

run(testDialectDescriptor)


# CHECK-LABEL: TEST: testUserDialectClass
def testUserDialectClass():
  ctx = mlir.ir.Context()
  # Access using attribute.
  d = ctx.dialects.std
  # Note that the standard dialect namespace prints as ''. Others will print
  # as "<Dialect %namespace (..."
  # CHECK: <Dialect (class mlir.dialects.std._Dialect)>
  print(d)
  try:
    _ = ctx.dialects.not_existing
  except AttributeError:
    pass
  else:
    assert False, "Expected exception"

  # Access using index.
  d = ctx.dialects["std"]
  # CHECK: <Dialect (class mlir.dialects.std._Dialect)>
  print(d)
  try:
    _ = ctx.dialects["not_existing"]
  except IndexError:
    pass
  else:
    assert False, "Expected exception"

  # Using the 'd' alias.
  d = ctx.d["std"]
  # CHECK: <Dialect (class mlir.dialects.std._Dialect)>
  print(d)

run(testUserDialectClass)


# CHECK-LABEL: TEST: testCustomOpView
# This test uses the standard dialect AddFOp as an example of a user op.
# TODO: Op creation and access is still quite verbose: simplify this test as
# additional capabilities come online.
def testCustomOpView():
  ctx = mlir.ir.Context()
  ctx.allow_unregistered_dialects = True
  f32 = mlir.ir.F32Type.get(ctx)
  loc = ctx.get_unknown_location()
  m = ctx.create_module(loc)
  m_block = m.operation.regions[0].blocks[0]
  # TODO: Remove integer insertion in favor of InsertionPoint and/or op-based.
  ip = [0]
  def createInput():
    op = ctx.create_operation("pytest_dummy.intinput", loc, results=[f32])
    m_block.operations.insert(ip[0], op)
    ip[0] += 1
    # TODO: Auto result cast from operation
    return op.results[0]

  # Create via dialects context collection.
  input1 = createInput()
  input2 = createInput()
  op1 = ctx.dialects.std.AddFOp(loc, input1, input2)
  # TODO: Auto operation cast from OpView
  # TODO: Context manager insertion point
  m_block.operations.insert(ip[0], op1.operation)
  ip[0] += 1

  # Create via an import
  from mlir.dialects.std import AddFOp
  op2 = AddFOp(loc, input1, op1.result)
  m_block.operations.insert(ip[0], op2.operation)
  ip[0] += 1

  # CHECK: %[[INPUT0:.*]] = "pytest_dummy.intinput"
  # CHECK: %[[INPUT1:.*]] = "pytest_dummy.intinput"
  # CHECK: %[[R0:.*]] = addf %[[INPUT0]], %[[INPUT1]] : f32
  # CHECK: %[[R1:.*]] = addf %[[INPUT0]], %[[R0]] : f32
  m.operation.print()

run(testCustomOpView)
