# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


# CHECK-LABEL: TEST: test_exception
@run
def test_exception():
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    try:
        Operation.parse(
            """
      func.func @foo() {
          "test.use"(%0) : (i64) -> ()  loc("use")
          %0 = "test.def"() : () -> i64 loc("def")
          return
      }
    """,
            context=ctx,
        )
    except MLIRError as e:
        # CHECK: Exception: <
        # CHECK:   Unable to parse operation assembly:
        # CHECK:   error: "use": operand #0 does not dominate this use
        # CHECK:    note: "use": see current operation: "test.use"(%0) : (i64) -> ()
        # CHECK:    note: "def": operand defined here (op in the same block)
        # CHECK: >
        print(f"Exception: <{e}>")

        # CHECK: message: Unable to parse operation assembly
        print(f"message: {e.message}")

        # CHECK: error_diagnostics[0]:           loc("use") operand #0 does not dominate this use
        # CHECK: error_diagnostics[0].notes[0]:  loc("use") see current operation: "test.use"(%0) : (i64) -> ()
        # CHECK: error_diagnostics[0].notes[1]:  loc("def") operand defined here (op in the same block)
        print(
            "error_diagnostics[0]:          ",
            e.error_diagnostics[0].location,
            e.error_diagnostics[0].message,
        )
        print(
            "error_diagnostics[0].notes[0]: ",
            e.error_diagnostics[0].notes[0].location,
            e.error_diagnostics[0].notes[0].message,
        )
        print(
            "error_diagnostics[0].notes[1]: ",
            e.error_diagnostics[0].notes[1].location,
            e.error_diagnostics[0].notes[1].message,
        )


# CHECK-LABEL: test_emit_error_diagnostics
@run
def test_emit_error_diagnostics():
    ctx = Context()
    loc = Location.unknown(ctx)
    handler_diags = []

    def handler(d):
        handler_diags.append(str(d))
        return True

    ctx.attach_diagnostic_handler(handler)

    try:
        Attribute.parse("not an attr", ctx)
    except MLIRError as e:
        # CHECK: emit_error_diagnostics=False:
        # CHECK: e.error_diagnostics: ['expected attribute value']
        # CHECK: handler_diags: []
        print(f"emit_error_diagnostics=False:")
        print(f"e.error_diagnostics: {[str(diag) for diag in e.error_diagnostics]}")
        print(f"handler_diags: {handler_diags}")

    ctx.emit_error_diagnostics = True
    try:
        Attribute.parse("not an attr", ctx)
    except MLIRError as e:
        # CHECK: emit_error_diagnostics=True:
        # CHECK: e.error_diagnostics: []
        # CHECK: handler_diags: ['expected attribute value']
        print(f"emit_error_diagnostics=True:")
        print(f"e.error_diagnostics: {[str(diag) for diag in e.error_diagnostics]}")
        print(f"handler_diags: {handler_diags}")
