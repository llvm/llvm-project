# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import ml_program


def constructAndPrintInModule(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      f()
    print(module)
  return f


# CHECK-LABEL: testFuncOp
@constructAndPrintInModule
def testFuncOp():
  # CHECK: ml_program.func @foobar(%arg0: si32) -> si32
  f = ml_program.FuncOp(
      name="foobar",
      type=([IntegerType.get_signed(32)], [IntegerType.get_signed(32)]))
  block = f.add_entry_block()
  with InsertionPoint(block):
    # CHECK: ml_program.return
    ml_program.ReturnOp([block.arguments[0]])
