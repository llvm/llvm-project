# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testUnknown
def testUnknown():
  with Context() as ctx:
    loc = Location.unknown()
  assert loc.context is ctx
  ctx = None
  gc.collect()
  # CHECK: unknown str: loc(unknown)
  print("unknown str:", str(loc))
  # CHECK: unknown repr: loc(unknown)
  print("unknown repr:", repr(loc))

run(testUnknown)


# CHECK-LABEL: TEST: testFileLineCol
def testFileLineCol():
  with Context() as ctx:
    loc = Location.file("foo.txt", 123, 56)
  ctx = None
  gc.collect()
  # CHECK: file str: loc("foo.txt":123:56)
  print("file str:", str(loc))
  # CHECK: file repr: loc("foo.txt":123:56)
  print("file repr:", repr(loc))

run(testFileLineCol)


# CHECK-LABEL: TEST: testLocationCapsule
def testLocationCapsule():
  with Context() as ctx:
    loc1 = Location.file("foo.txt", 123, 56)
  # CHECK: mlir.ir.Location._CAPIPtr
  loc_capsule = loc1._CAPIPtr
  print(loc_capsule)
  loc2 = Location._CAPICreate(loc_capsule)
  assert loc2 == loc1
  assert loc2.context is ctx

run(testLocationCapsule)
