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

