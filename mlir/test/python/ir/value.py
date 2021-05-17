# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testCapsuleConversions
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


run(testCapsuleConversions)


# CHECK-LABEL: TEST: testOpResultOwner
def testOpResultOwner():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    i32 = IntegerType.get_signless(32)
    op = Operation.create("custom.op1", results=[i32])
    assert op.result.owner == op


run(testOpResultOwner)
