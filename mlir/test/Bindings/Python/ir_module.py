# RUN: %PYTHON %s | FileCheck %s

import gc
import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert mlir.ir.Context._get_live_count() == 0


# Verify successful parse.
# CHECK-LABEL: TEST: testParseSuccess
# CHECK: module @successfulParse
def testParseSuccess():
  ctx = mlir.ir.Context()
  module = ctx.parse_module(r"""module @successfulParse {}""")
  print("CLEAR CONTEXT")
  ctx = None  # Ensure that module captures the context.
  gc.collect()
  module.dump()  # Just outputs to stderr. Verifies that it functions.
  print(str(module))

run(testParseSuccess)


# Verify parse error.
# CHECK-LABEL: TEST: testParseError
# CHECK: testParseError: Unable to parse module assembly (see diagnostics)
def testParseError():
  ctx = mlir.ir.Context()
  try:
    module = ctx.parse_module(r"""}SYNTAX ERROR{""")
  except ValueError as e:
    print("testParseError:", e)
  else:
    print("Exception not produced")

run(testParseError)


# Verify round-trip of ASM that contains unicode.
# Note that this does not test that the print path converts unicode properly
# because MLIR asm always normalizes it to the hex encoding.
# CHECK-LABEL: TEST: testRoundtripUnicode
# CHECK: func @roundtripUnicode()
# CHECK: foo = "\F0\9F\98\8A"
def testRoundtripUnicode():
  ctx = mlir.ir.Context()
  module = ctx.parse_module(r"""
    func @roundtripUnicode() attributes { foo = "😊" }
  """)
  print(str(module))

run(testRoundtripUnicode)


# Tests that module.operation works and correctly interns instances.
# CHECK-LABEL: TEST: testModuleOperation
def testModuleOperation():
  ctx = mlir.ir.Context()
  module = ctx.parse_module(r"""module @successfulParse {}""")
  op1 = module.operation
  assert ctx._get_live_operation_count() == 1
  # CHECK: module @successfulParse
  print(op1)

  # Ensure that operations are the same on multiple calls.
  op2 = module.operation
  assert ctx._get_live_operation_count() == 1
  assert op1 is op2

  # Ensure that if module is de-referenced, the operations are still valid.
  module = None
  gc.collect()
  print(op1)

  # Collect and verify lifetime.
  op1 = None
  op2 = None
  gc.collect()
  print("LIVE OPERATIONS:", ctx._get_live_operation_count())
  assert ctx._get_live_operation_count() == 0

run(testModuleOperation)
