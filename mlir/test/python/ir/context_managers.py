# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testContextEnterExit
def testContextEnterExit():
  with Context() as ctx:
    assert Context.current is ctx
  try:
    _ = Context.current
  except ValueError as e:
    # CHECK: No current Context
    print(e)
  else: assert False, "Expected exception"

run(testContextEnterExit)


# CHECK-LABEL: TEST: testLocationEnterExit
def testLocationEnterExit():
  ctx1 = Context()
  with Location.unknown(ctx1) as loc1:
    assert Context.current is ctx1
    assert Location.current is loc1

    # Re-asserting the same context should not change the location.
    with ctx1:
      assert Context.current is ctx1
      assert Location.current is loc1
      # Asserting a different context should clear it.
      with Context() as ctx2:
        assert Context.current is ctx2
        try:
          _ = Location.current
        except ValueError: pass
        else: assert False, "Expected exception"

      # And should restore.
      assert Context.current is ctx1
      assert Location.current is loc1

  # All should clear.
  try:
    _ = Location.current
  except ValueError as e:
    # CHECK: No current Location
    print(e)
  else: assert False, "Expected exception"

run(testLocationEnterExit)


# CHECK-LABEL: TEST: testInsertionPointEnterExit
def testInsertionPointEnterExit():
  ctx1 = Context()
  m = Module.create(Location.unknown(ctx1))
  ip = InsertionPoint(m.body)

  with ip:
    assert InsertionPoint.current is ip
    # Asserting a location from the same context should preserve.
    with Location.unknown(ctx1) as loc1:
      assert InsertionPoint.current is ip
      assert Location.current is loc1
    # Location should clear.
    try:
      _ = Location.current
    except ValueError: pass
    else: assert False, "Expected exception"

    # Asserting the same Context should preserve.
    with ctx1:
      assert InsertionPoint.current is ip

    # Asserting a different context should clear it.
    with Context() as ctx2:
      assert Context.current is ctx2
      try:
        _ = InsertionPoint.current
      except ValueError: pass
      else: assert False, "Expected exception"

  # All should clear.
  try:
    _ = InsertionPoint.current
  except ValueError as e:
    # CHECK: No current InsertionPoint
    print(e)
  else: assert False, "Expected exception"

run(testInsertionPointEnterExit)
