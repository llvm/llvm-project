# RUN: %PYTHON %s | FileCheck %s

import mlir

def run(f):
  print("\nTEST:", f.__name__)
  f()

# Verify successful parse.
# CHECK-LABEL: TEST: testParseSuccess
# CHECK: module @successfulParse
def testParseSuccess():
  ctx = mlir.ir.Context()
  module = ctx.parse_module(r"""module @successfulParse {}""")
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
