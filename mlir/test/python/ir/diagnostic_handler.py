# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


@run
def testLifecycleContextDestroy():
    ctx = Context()

    def callback(foo):
        ...

    handler = ctx.attach_diagnostic_handler(callback)
    assert handler.attached
    # If context is destroyed before the handler, it should auto-detach.
    ctx = None
    gc.collect()
    assert not handler.attached

    # And finally collecting the handler should be fine.
    handler = None
    gc.collect()


@run
def testLifecycleExplicitDetach():
    ctx = Context()

    def callback(foo):
        ...

    handler = ctx.attach_diagnostic_handler(callback)
    assert handler.attached
    handler.detach()
    assert not handler.attached


@run
def testLifecycleWith():
    ctx = Context()

    def callback(foo):
        ...

    with ctx.attach_diagnostic_handler(callback) as handler:
        assert handler.attached
    assert not handler.attached


@run
def testLifecycleWithAndExplicitDetach():
    ctx = Context()

    def callback(foo):
        ...

    with ctx.attach_diagnostic_handler(callback) as handler:
        assert handler.attached
        handler.detach()
    assert not handler.attached


# CHECK-LABEL: TEST: testDiagnosticCallback
@run
def testDiagnosticCallback():
    ctx = Context()

    def callback(d):
        # CHECK: DIAGNOSTIC: message='foobar', severity=DiagnosticSeverity.ERROR, loc=loc(unknown)
        print(
            f"DIAGNOSTIC: message='{d.message}', severity={d.severity}, loc={d.location}"
        )
        return True

    handler = ctx.attach_diagnostic_handler(callback)
    loc = Location.unknown(ctx)
    loc.emit_error("foobar")
    assert not handler.had_error


# CHECK-LABEL: TEST: testDiagnosticEmptyNotes
# TODO: Come up with a way to inject a diagnostic with notes from this API.
@run
def testDiagnosticEmptyNotes():
    ctx = Context()

    def callback(d):
        # CHECK: DIAGNOSTIC: notes=()
        print(f"DIAGNOSTIC: notes={d.notes}")
        return True

    handler = ctx.attach_diagnostic_handler(callback)
    loc = Location.unknown(ctx)
    loc.emit_error("foobar")
    assert not handler.had_error


# CHECK-LABEL: TEST: testDiagnosticNonEmptyNotes
@run
def testDiagnosticNonEmptyNotes():
    ctx = Context()
    ctx.emit_error_diagnostics = True

    def callback(d):
        # CHECK: DIAGNOSTIC:
        # CHECK:   message='arith.addi' op requires one result
        # CHECK:   notes=['see current operation: "arith.addi"() {{.*}} : () -> ()']
        print(f"DIAGNOSTIC:")
        print(f"  message={d.message}")
        print(f"  notes={list(map(str, d.notes))}")
        return True

    handler = ctx.attach_diagnostic_handler(callback)
    loc = Location.unknown(ctx)
    try:
        Operation.create("arith.addi", loc=loc).verify()
    except MLIRError:
        pass
    assert not handler.had_error


# CHECK-LABEL: TEST: testDiagnosticCallbackException
@run
def testDiagnosticCallbackException():
    ctx = Context()

    def callback(d):
        raise ValueError("Error in handler")

    handler = ctx.attach_diagnostic_handler(callback)
    loc = Location.unknown(ctx)
    loc.emit_error("foobar")
    assert handler.had_error


# CHECK-LABEL: TEST: testEscapingDiagnostic
@run
def testEscapingDiagnostic():
    ctx = Context()
    diags = []

    def callback(d):
        diags.append(d)
        return True

    handler = ctx.attach_diagnostic_handler(callback)
    loc = Location.unknown(ctx)
    loc.emit_error("foobar")
    assert not handler.had_error

    # CHECK: DIAGNOSTIC: <Invalid Diagnostic>
    print(f"DIAGNOSTIC: {str(diags[0])}")
    try:
        diags[0].severity
        raise RuntimeError("expected exception")
    except ValueError:
        pass
    try:
        diags[0].location
        raise RuntimeError("expected exception")
    except ValueError:
        pass
    try:
        diags[0].message
        raise RuntimeError("expected exception")
    except ValueError:
        pass
    try:
        diags[0].notes
        raise RuntimeError("expected exception")
    except ValueError:
        pass


# CHECK-LABEL: TEST: testDiagnosticReturnTrueHandles
@run
def testDiagnosticReturnTrueHandles():
    ctx = Context()

    def callback1(d):
        print(f"CALLBACK1: {d}")
        return True

    def callback2(d):
        print(f"CALLBACK2: {d}")
        return True

    ctx.attach_diagnostic_handler(callback1)
    ctx.attach_diagnostic_handler(callback2)
    loc = Location.unknown(ctx)
    # CHECK-NOT: CALLBACK1
    # CHECK: CALLBACK2: foobar
    # CHECK-NOT: CALLBACK1
    loc.emit_error("foobar")


# CHECK-LABEL: TEST: testDiagnosticReturnFalseDoesNotHandle
@run
def testDiagnosticReturnFalseDoesNotHandle():
    ctx = Context()

    def callback1(d):
        print(f"CALLBACK1: {d}")
        return True

    def callback2(d):
        print(f"CALLBACK2: {d}")
        return False

    ctx.attach_diagnostic_handler(callback1)
    ctx.attach_diagnostic_handler(callback2)
    loc = Location.unknown(ctx)
    # CHECK: CALLBACK2: foobar
    # CHECK: CALLBACK1: foobar
    loc.emit_error("foobar")
