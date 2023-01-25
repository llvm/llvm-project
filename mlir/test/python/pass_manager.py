# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc, sys
from mlir.ir import *
from mlir.passmanager import *

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()

def run(f):
  log("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0

# Verify capsule interop.
# CHECK-LABEL: TEST: testCapsule
def testCapsule():
  with Context():
    pm = PassManager()
    pm_capsule = pm._CAPIPtr
    assert '"mlir.passmanager.PassManager._CAPIPtr"' in repr(pm_capsule)
    pm._testing_release()
    pm1 = PassManager._CAPICreate(pm_capsule)
    assert pm1 is not None  # And does not crash.
run(testCapsule)

# CHECK-LABEL: TEST: testConstruct
@run
def testConstruct():
  with Context():
    # CHECK: pm1: 'any()'
    # CHECK: pm2: 'builtin.module()'
    pm1 = PassManager()
    pm2 = PassManager("builtin.module")
    log(f"pm1: '{pm1}'")
    log(f"pm2: '{pm2}'")


# Verify successful round-trip.
# CHECK-LABEL: TEST: testParseSuccess
def testParseSuccess():
  with Context():
    # An unregistered pass should not parse.
    try:
      pm = PassManager.parse("builtin.module(func.func(not-existing-pass{json=false}))")
    except ValueError as e:
      # CHECK: ValueError exception: {{.+}} 'not-existing-pass' does not refer to a registered pass
      log("ValueError exception:", e)
    else:
      log("Exception not produced")

    # A registered pass should parse successfully.
    pm = PassManager.parse("builtin.module(func.func(print-op-stats{json=false}))")
    # CHECK: Roundtrip: builtin.module(func.func(print-op-stats{json=false}))
    log("Roundtrip: ", pm)
run(testParseSuccess)

# Verify failure on unregistered pass.
# CHECK-LABEL: TEST: testParseFail
def testParseFail():
  with Context():
    try:
      pm = PassManager.parse("any(unknown-pass)")
    except ValueError as e:
      #      CHECK: ValueError exception: MLIR Textual PassPipeline Parser:1:1: error:
      # CHECK-SAME: 'unknown-pass' does not refer to a registered pass or pass pipeline
      #      CHECK: unknown-pass
      #      CHECK: ^
      log("ValueError exception:", e)
    else:
      log("Exception not produced")
run(testParseFail)

# Check that adding to a pass manager works
# CHECK-LABEL: TEST: testAdd
@run
def testAdd():
  pm = PassManager("any", Context())
  # CHECK: pm: 'any()'
  log(f"pm: '{pm}'")
  # CHECK: pm: 'any(cse)'
  pm.add("cse")
  log(f"pm: '{pm}'")
  # CHECK: pm: 'any(cse,cse)'
  pm.add("cse")
  log(f"pm: '{pm}'")


# Verify failure on incorrect level of nesting.
# CHECK-LABEL: TEST: testInvalidNesting
def testInvalidNesting():
  with Context():
    try:
      pm = PassManager.parse("func.func(normalize-memrefs)")
    except ValueError as e:
      # CHECK: ValueError exception: Can't add pass 'NormalizeMemRefs' restricted to 'builtin.module' on a PassManager intended to run on 'func.func', did you intend to nest?
      log("ValueError exception:", e)
    else:
      log("Exception not produced")
run(testInvalidNesting)


# Verify that a pass manager can execute on IR
# CHECK-LABEL: TEST: testRun
def testRunPipeline():
  with Context():
    pm = PassManager.parse("builtin.module(print-op-stats{json=false})")
    module = Module.parse(r"""func.func @successfulParse() { return }""")
    pm.run(module)
# CHECK: Operations encountered:
# CHECK: builtin.module    , 1
# CHECK: func.func      , 1
# CHECK: func.return        , 1
run(testRunPipeline)
