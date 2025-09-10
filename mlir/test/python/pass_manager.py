# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc, os, sys, tempfile
from mlir.ir import *
from mlir.passmanager import *
from mlir.dialects.func import FuncOp
from mlir.dialects.builtin import ModuleOp


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
            pm = PassManager.parse(
                "builtin.module(func.func(not-existing-pass{json=false}))"
            )
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


# Verify successful round-trip.
# CHECK-LABEL: TEST: testParseSpacedPipeline
def testParseSpacedPipeline():
    with Context():
        # A registered pass should parse successfully even if has extras spaces for readability
        pm = PassManager.parse(
            """builtin.module(
        func.func( print-op-stats{ json=false } )
    )"""
        )
        # CHECK: Roundtrip: builtin.module(func.func(print-op-stats{json=false}))
        log("Roundtrip: ", pm)


run(testParseSpacedPipeline)


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
            # CHECK: ValueError exception: Can't add pass 'NormalizeMemRefsPass' restricted to 'builtin.module' on a PassManager intended to run on 'func.func', did you intend to nest?
            log("ValueError exception:", e)
        else:
            log("Exception not produced")


run(testInvalidNesting)


# Verify that a pass manager can execute on IR
# CHECK-LABEL: TEST: testRunPipeline
def testRunPipeline():
    with Context():
        pm = PassManager.parse("any(print-op-stats{json=false})")
        func = FuncOp.parse(r"""func.func @successfulParse() { return }""")
        pm.run(func)


# CHECK: Operations encountered:
# CHECK: func.func      , 1
# CHECK: func.return        , 1
run(testRunPipeline)


# CHECK-LABEL: TEST: testRunPipelineError
@run
def testRunPipelineError():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        op = Operation.parse('"test.op"() : () -> ()')
        pm = PassManager.parse("any(cse)")
        try:
            pm.run(op)
        except MLIRError as e:
            # CHECK: Exception: <
            # CHECK:   Failure while executing pass pipeline:
            # CHECK:   error: "-":1:1: 'test.op' op trying to schedule a pass on an unregistered operation
            # CHECK:    note: "-":1:1: see current operation: "test.op"() : () -> ()
            # CHECK: >
            log(f"Exception: <{e}>")


# CHECK-LABEL: TEST: testPostPassOpInvalidation
@run
def testPostPassOpInvalidation():
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            arith.constant 10
            func.func @foo() {
              arith.constant 10
              return
            }
          }
        """
        )

        outer_const_op = module.body.operations[0]
        # CHECK: %[[VAL0:.*]] = arith.constant 10 : i64
        log(outer_const_op)

        func_op = module.body.operations[1]
        # CHECK: func.func @[[FOO:.*]]() {
        # CHECK:   %[[VAL1:.*]] = arith.constant 10 : i64
        # CHECK:   return
        # CHECK: }
        log(func_op)

        inner_const_op = func_op.body.blocks[0].operations[0]
        # CHECK: %[[VAL1]] = arith.constant 10 : i64
        log(inner_const_op)

        PassManager.parse("builtin.module(canonicalize)").run(module)
        # CHECK: func.func @foo() {
        # CHECK:   return
        # CHECK: }
        log(func_op)

        # CHECK: func.func @foo() {
        # CHECK:   return
        # CHECK: }
        log(module)

        # CHECK: invalidate_ops=True
        log("invalidate_ops=True")

        module = ModuleOp.parse(
            """
          module {
            arith.constant 10
            func.func @foo() {
              arith.constant 10
              return
            }
          }
        """
        )

        PassManager.parse("builtin.module(canonicalize)").run(module)

        func_op._set_invalid()
        try:
            log(func_op)
        except RuntimeError as e:
            # CHECK: the operation has been invalidated
            log(e)

        outer_const_op._set_invalid()
        try:
            log(outer_const_op)
        except RuntimeError as e:
            # CHECK: the operation has been invalidated
            log(e)

        inner_const_op._set_invalid()
        try:
            log(inner_const_op)
        except RuntimeError as e:
            # CHECK: the operation has been invalidated
            log(e)

        # CHECK: func.func @foo() {
        # CHECK:   return
        # CHECK: }
        log(module)


# CHECK-LABEL: TEST: testPrintIrAfterAll
@run
def testPrintIrAfterAll():
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            func.func @main() {
              %0 = arith.constant 10
              return
            }
          }
        """
        )
        pm = PassManager.parse("builtin.module(canonicalize)")
        ctx.enable_multithreading(False)
        pm.enable_ir_printing()
        # CHECK: // -----// IR Dump After Canonicalizer (canonicalize) //----- //
        # CHECK: module {
        # CHECK:   func.func @main() {
        # CHECK:     return
        # CHECK:   }
        # CHECK: }
        pm.run(module)


# CHECK-LABEL: TEST: testPrintIrBeforeAndAfterAll
@run
def testPrintIrBeforeAndAfterAll():
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            func.func @main() {
              %0 = arith.constant 10
              return
            }
          }
        """
        )
        pm = PassManager.parse("builtin.module(canonicalize)")
        ctx.enable_multithreading(False)
        pm.enable_ir_printing(print_before_all=True, print_after_all=True)
        # CHECK: // -----// IR Dump Before Canonicalizer (canonicalize) //----- //
        # CHECK: module {
        # CHECK:   func.func @main() {
        # CHECK:     %[[C10:.*]] = arith.constant 10 : i64
        # CHECK:     return
        # CHECK:   }
        # CHECK: }
        # CHECK: // -----// IR Dump After Canonicalizer (canonicalize) //----- //
        # CHECK: module {
        # CHECK:   func.func @main() {
        # CHECK:     return
        # CHECK:   }
        # CHECK: }
        pm.run(module)


# CHECK-LABEL: TEST: testPrintIrLargeLimitElements
@run
def testPrintIrLargeLimitElements():
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            func.func @main() -> tensor<3xi64> {
              %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi64>
              return %0 : tensor<3xi64>
            }
          }
        """
        )
        pm = PassManager.parse("builtin.module(canonicalize)")
        ctx.enable_multithreading(False)
        pm.enable_ir_printing(large_elements_limit=2)
        # CHECK:     %[[CST:.*]] = arith.constant dense_resource<__elided__> : tensor<3xi64>
        pm.run(module)


# CHECK-LABEL: TEST: testPrintIrLargeResourceLimit
@run
def testPrintIrLargeResourceLimit():
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            func.func @main() -> tensor<3xi64> {
              %0 = arith.constant dense_resource<blob1> : tensor<3xi64>
              return %0 : tensor<3xi64>
            }
          }
          {-#
            dialect_resources: {
              builtin: {
                blob1: "0x010000000000000002000000000000000300000000000000"
              }
            }
          #-}
        """
        )
        pm = PassManager.parse("builtin.module(canonicalize)")
        ctx.enable_multithreading(False)
        pm.enable_ir_printing(large_resource_limit=4)
        # CHECK-NOT: blob1: "0x01
        pm.run(module)


# CHECK-LABEL: TEST: testPrintIrLargeResourceLimitVsElementsLimit
@run
def testPrintIrLargeResourceLimitVsElementsLimit():
    """Test that large_elements_limit does not affect the printing of resources."""
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            func.func @main() -> tensor<3xi64> {
              %0 = arith.constant dense_resource<blob1> : tensor<3xi64>
              return %0 : tensor<3xi64>
            }
          }
          {-#
            dialect_resources: {
              builtin: {
                blob1: "0x010000000000000002000000000000000300000000000000"
              }
            }
          #-}
        """
        )
        pm = PassManager.parse("builtin.module(canonicalize)")
        ctx.enable_multithreading(False)
        pm.enable_ir_printing(large_elements_limit=1)
        # CHECK-NOT: blob1: "0x01
        pm.run(module)


# CHECK-LABEL: TEST: testPrintIrTree
@run
def testPrintIrTree():
    with Context() as ctx:
        module = ModuleOp.parse(
            """
          module {
            func.func @main() {
              %0 = arith.constant 10
              return
            }
          }
        """
        )
        pm = PassManager.parse("builtin.module(canonicalize)")
        ctx.enable_multithreading(False)
        pm.enable_ir_printing()
        # CHECK-LABEL: // Tree printing begin
        # CHECK: \-- builtin_module_no-symbol-name
        # CHECK:     \-- 0_canonicalize.mlir
        # CHECK-LABEL: // Tree printing end
        pm.run(module)
        log("// Tree printing begin")
        with tempfile.TemporaryDirectory() as temp_dir:
            pm.enable_ir_printing(tree_printing_dir_path=temp_dir)
            pm.run(module)

            def print_file_tree(directory, prefix=""):
                entries = sorted(os.listdir(directory))
                for i, entry in enumerate(entries):
                    path = os.path.join(directory, entry)
                    connector = "\-- " if i == len(entries) - 1 else "|-- "
                    log(f"{prefix}{connector}{entry}")
                    if os.path.isdir(path):
                        print_file_tree(
                            path, prefix + ("    " if i == len(entries) - 1 else "│   ")
                        )

            print_file_tree(temp_dir)
        log("// Tree printing end")
