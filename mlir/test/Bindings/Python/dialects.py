# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testDialectDescriptor
def testDialectDescriptor():
  ctx = Context()
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
  ctx = Context()
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
  def createInput():
    op = Operation.create("pytest_dummy.intinput", results=[f32])
    # TODO: Auto result cast from operation
    return op.results[0]

  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    m = Module.create()

    with InsertionPoint.at_block_terminator(m.body):
      f32 = F32Type.get()
      # Create via dialects context collection.
      input1 = createInput()
      input2 = createInput()
      op1 = ctx.dialects.std.AddFOp(input1.type, input1, input2)

      # Create via an import
      from mlir.dialects.std import AddFOp
      AddFOp(input1.type, input1, op1.result)

  # CHECK: %[[INPUT0:.*]] = "pytest_dummy.intinput"
  # CHECK: %[[INPUT1:.*]] = "pytest_dummy.intinput"
  # CHECK: %[[R0:.*]] = addf %[[INPUT0]], %[[INPUT1]] : f32
  # CHECK: %[[R1:.*]] = addf %[[INPUT0]], %[[R0]] : f32
  m.operation.print()


run(testCustomOpView)
