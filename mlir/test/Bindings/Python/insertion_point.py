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


# CHECK-LABEL: TEST: test_insert_at_block_end
def test_insert_at_block_end():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    module = Module.parse(r"""
      func @foo() -> () {
        "custom.op1"() : () -> ()
      }
    """)
    entry_block = module.body.operations[0].regions[0].blocks[0]
    ip = InsertionPoint(entry_block)
    ip.insert(Operation.create("custom.op2"))
    # CHECK: "custom.op1"
    # CHECK: "custom.op2"
    module.operation.print()

run(test_insert_at_block_end)


# CHECK-LABEL: TEST: test_insert_before_operation
def test_insert_before_operation():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    module = Module.parse(r"""
      func @foo() -> () {
        "custom.op1"() : () -> ()
        "custom.op2"() : () -> ()
      }
    """)
    entry_block = module.body.operations[0].regions[0].blocks[0]
    ip = InsertionPoint(entry_block.operations[1])
    ip.insert(Operation.create("custom.op3"))
    # CHECK: "custom.op1"
    # CHECK: "custom.op3"
    # CHECK: "custom.op2"
    module.operation.print()

run(test_insert_before_operation)


# CHECK-LABEL: TEST: test_insert_at_block_begin
def test_insert_at_block_begin():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    module = Module.parse(r"""
      func @foo() -> () {
        "custom.op2"() : () -> ()
      }
    """)
    entry_block = module.body.operations[0].regions[0].blocks[0]
    ip = InsertionPoint.at_block_begin(entry_block)
    ip.insert(Operation.create("custom.op1"))
    # CHECK: "custom.op1"
    # CHECK: "custom.op2"
    module.operation.print()

run(test_insert_at_block_begin)


# CHECK-LABEL: TEST: test_insert_at_block_begin_empty
def test_insert_at_block_begin_empty():
  # TODO: Write this test case when we can create such a situation.
  pass

run(test_insert_at_block_begin_empty)


# CHECK-LABEL: TEST: test_insert_at_terminator
def test_insert_at_terminator():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    module = Module.parse(r"""
      func @foo() -> () {
        "custom.op1"() : () -> ()
        return
      }
    """)
    entry_block = module.body.operations[0].regions[0].blocks[0]
    ip = InsertionPoint.at_block_terminator(entry_block)
    ip.insert(Operation.create("custom.op2"))
    # CHECK: "custom.op1"
    # CHECK: "custom.op2"
    module.operation.print()

run(test_insert_at_terminator)


# CHECK-LABEL: TEST: test_insert_at_block_terminator_missing
def test_insert_at_block_terminator_missing():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with ctx:
    module = Module.parse(r"""
      func @foo() -> () {
        "custom.op1"() : () -> ()
      }
    """)
    entry_block = module.body.operations[0].regions[0].blocks[0]
    try:
      ip = InsertionPoint.at_block_terminator(entry_block)
    except ValueError as e:
      # CHECK: Block has no terminator
      print(e)
    else:
      assert False, "Expected exception"

run(test_insert_at_block_terminator_missing)


# CHECK-LABEL: TEST: test_insert_at_end_with_terminator_errors
def test_insert_at_end_with_terminator_errors():
  with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    m = Module.create()  # Module is created with a terminator.
    with InsertionPoint(m.body):
      try:
        Operation.create("custom.op1", results=[], operands=[])
      except IndexError as e:
        # CHECK: ERROR: Cannot insert operation at the end of a block that already has a terminator.
        print(f"ERROR: {e}")

run(test_insert_at_end_with_terminator_errors)


# CHECK-LABEL: TEST: test_insertion_point_context
def test_insertion_point_context():
  ctx = Context()
  ctx.allow_unregistered_dialects = True
  with Location.unknown(ctx):
    module = Module.parse(r"""
      func @foo() -> () {
        "custom.op1"() : () -> ()
      }
    """)
    entry_block = module.body.operations[0].regions[0].blocks[0]
    with InsertionPoint(entry_block):
      Operation.create("custom.op2")
      with InsertionPoint.at_block_begin(entry_block):
        Operation.create("custom.opa")
        Operation.create("custom.opb")
      Operation.create("custom.op3")
    # CHECK: "custom.opa"
    # CHECK: "custom.opb"
    # CHECK: "custom.op1"
    # CHECK: "custom.op2"
    # CHECK: "custom.op3"
    module.operation.print()

run(test_insertion_point_context)
