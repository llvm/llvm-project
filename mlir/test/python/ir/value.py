# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0
  return f


# CHECK-LABEL: TEST: testCapsuleConversions
@run
def testCapsuleConversions():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    i32 = IntegerType.get_signless(32)
    value = Operation.create("custom.op1", results=[i32]).result
    value_capsule = value._CAPIPtr
    assert '"mlir.ir.Value._CAPIPtr"' in repr(value_capsule)
    value2 = Value._CAPICreate(value_capsule)
    assert value2 == value


# CHECK-LABEL: TEST: testOpResultOwner
@run
def testOpResultOwner():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    i32 = IntegerType.get_signless(32)
    op = Operation.create("custom.op1", results=[i32])
    assert op.result.owner == op


# CHECK-LABEL: TEST: testValueIsInstance
@run
def testValueIsInstance():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(
      r"""
    func.func @foo(%arg0: f32) {
      %0 = "some_dialect.some_op"() : () -> f64
      return
    }""", ctx)
  func = module.body.operations[0]
  assert BlockArgument.isinstance(func.regions[0].blocks[0].arguments[0])
  assert not OpResult.isinstance(func.regions[0].blocks[0].arguments[0])

  op = func.regions[0].blocks[0].operations[0]
  assert not BlockArgument.isinstance(op.results[0])
  assert OpResult.isinstance(op.results[0])


# CHECK-LABEL: TEST: testValueHash
@run
def testValueHash():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  module = Module.parse(
      r"""
    func.func @foo(%arg0: f32) -> f32 {
      %0 = "some_dialect.some_op"(%arg0) : (f32) -> f32
      return %0 : f32
    }""", ctx)

  [func] = module.body.operations
  block = func.entry_block
  op, ret = block.operations
  assert hash(block.arguments[0]) == hash(op.operands[0])
  assert hash(op.result) == hash(ret.operands[0])
