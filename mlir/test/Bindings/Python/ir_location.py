# RUN: %PYTHON %s | FileCheck %s

import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()

# CHECK-LABEL: TEST: testUnknown
def testUnknown():
  ctx = mlir.ir.Context()
  loc = ctx.get_unknown_location()
  # CHECK: unknown str: loc(unknown)
  print("unknown str:", str(loc))
  # CHECK: unknown repr: loc(unknown)
  print("unknown repr:", repr(loc))

run(testUnknown)


# CHECK-LABEL: TEST: testFileLineCol
def testFileLineCol():
  ctx = mlir.ir.Context()
  loc = ctx.get_file_location("foo.txt", 123, 56)
  # CHECK: file str: loc("foo.txt":123:56)
  print("file str:", str(loc))
  # CHECK: file repr: loc("foo.txt":123:56)
  print("file repr:", repr(loc))

run(testFileLineCol)

