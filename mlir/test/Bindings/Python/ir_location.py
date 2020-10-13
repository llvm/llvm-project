# RUN: %PYTHON %s | FileCheck %s

import gc
import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0


# CHECK-LABEL: TEST: testUnknown
def testUnknown():
  ctx = mlir.ir.Context()
  loc = ctx.get_unknown_location()
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
  ctx = mlir.ir.Context()
  loc = ctx.get_file_location("foo.txt", 123, 56)
  ctx = None
  gc.collect()
  # CHECK: file str: loc("foo.txt":123:56)
  print("file str:", str(loc))
  # CHECK: file repr: loc("foo.txt":123:56)
  print("file repr:", repr(loc))

run(testFileLineCol)

