# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *

def run(f):
  print("\nTEST:", f.__name__)
  f()


# CHECK-LABEL: TEST: testNameIsPrivate
def testNameIsPrivate():
  # `import *` ignores private names starting with an understore, so the debug
  # flag shouldn't be visible unless explicitly imported.
  try:
    _GlobalDebug.flag = True
  except NameError:
    pass
  else:
    assert False, "_GlobalDebug must not be available by default"

run(testNameIsPrivate)


# CHECK-LABEL: TEST: testDebugDlag
def testDebugDlag():
  # Private names must be imported expilcitly.
  from mlir.ir import _GlobalDebug

  # CHECK: False
  print(_GlobalDebug.flag)
  _GlobalDebug.flag = True
  # CHECK: True
  print(_GlobalDebug.flag)
  _GlobalDebug.flag = False
  # CHECK: False
  print(_GlobalDebug.flag)

run(testDebugDlag)

