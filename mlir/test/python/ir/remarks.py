# RUN: %PYTHON %s | FileCheck %s

import gc
import os
import tempfile
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0
    return f


def emit(loc, kind=RemarkKind.PASSED, name="Unroll", category="Loop", **kwargs):
    return loc.emit_remark(kind, name, category=category, **kwargs)


def collect(into, attribute="remark_name"):
    """Returns a remark callback appending `attribute` of each remark to `into`."""
    return lambda remark: into.append(getattr(remark, attribute))


# CHECK-LABEL: TEST: testNoEngine
@run
def testNoEngine():
    ctx = Context()
    loc = Location.file("test.mlir", 1, 2, context=ctx)
    # CHECK: enabled: False
    print("enabled:", ctx.remarks_enabled)
    # CHECK: emitted: False
    print("emitted:", emit(loc))
    # Finalizing without an engine is a no-op.
    ctx.finalize_remarks()
    # CHECK: still disabled: False
    print("still disabled:", ctx.remarks_enabled)


# CHECK-LABEL: TEST: testEnableFinalize
@run
def testEnableFinalize():
    ctx = Context()
    states = [ctx.remarks_enabled]
    ctx.enable_remarks(all_filter=".*")
    states.append(ctx.remarks_enabled)
    ctx.finalize_remarks()
    states.append(ctx.remarks_enabled)
    ctx.finalize_remarks()
    states.append(ctx.remarks_enabled)
    ctx.enable_remarks(policy=RemarkPolicy.FINAL, print_as_emit_remarks=False)
    states.append(ctx.remarks_enabled)
    ctx.finalize_remarks()
    states.append(ctx.remarks_enabled)
    # CHECK: states: [False, True, False, False, True, False]
    print("states:", states)


# CHECK-LABEL: TEST: testCallback
@run
def testCallback():
    ctx = Context()

    def callback(remark):
        print(
            "kind:",
            remark.kind,
            "name:",
            remark.remark_name,
            "category:",
            remark.category_name,
            "full:",
            remark.full_category_name,
            "function:",
            remark.function_name,
            "id:",
            remark.remark_id,
        )
        print("args:", remark.args)
        print("message:", remark.message)
        print("str:", str(remark))
        print(
            "location:",
            remark.location,
            type(remark.location).__name__,
            remark.location.context is ctx,
        )

    ctx.enable_remarks(all_filter=".*", callback=callback)
    # CHECK: enabled: True
    print("enabled:", ctx.remarks_enabled)
    loc = Location.file("test.mlir", 3, 4, context=ctx)
    # The arguments keep their insertion order (the engine's id, the message,
    # the explicit pairs); the message sorts them by key.
    # CHECK: kind: RemarkKind.PASSED name: Unroll category: Loop full: Loop:Inner function: main id: 1
    # CHECK: args: [('RemarkId', '1'), ('Remark', 'unrolled by 4'), ('factor', '4')]
    # CHECK: message: [Passed] Unroll | Category:Loop:Inner | Function=main | Remark="unrolled by 4", RemarkId=1, factor=4
    # CHECK: str: [Passed] Unroll | Category:Loop:Inner | Function=main | Remark="unrolled by 4", RemarkId=1, factor=4
    # CHECK: location: loc("test.mlir":3:4) FileLineColLoc True
    # CHECK: emitted: True
    emitted = emit(
        loc,
        sub_category="Inner",
        function_name="main",
        message="unrolled by 4",
        args=[("factor", "4")],
    )
    print("emitted:", emitted)
    # Unset names fall back to placeholders; the only argument is the id.
    # CHECK: kind: RemarkKind.MISSED name: <unknown remark name> category:  full:  function: <unknown function> id: 2
    # CHECK: args: [('RemarkId', '2')]
    # CHECK: message: [Missed]  | RemarkId=2
    Location.unknown(context=ctx).emit_remark(RemarkKind.MISSED, "")
    ctx.finalize_remarks()
    # CHECK: enabled after finalize: False
    print("enabled after finalize:", ctx.remarks_enabled)


# CHECK-LABEL: TEST: testRemarkInvalidatedAfterCallback
@run
def testRemarkInvalidatedAfterCallback():
    ctx = Context()
    kept = []
    ctx.enable_remarks(all_filter=".*", callback=kept.append)
    emit(Location.unknown(context=ctx))
    ctx.finalize_remarks()
    remark = kept[0]
    for attribute in (
        "kind",
        "remark_name",
        "category_name",
        "full_category_name",
        "function_name",
        "location",
        "remark_id",
        "args",
        "message",
    ):
        try:
            getattr(remark, attribute)
            print(attribute, "-> no error (unexpected)")
        except ValueError as e:
            print(attribute, "->", e)
    # CHECK: kind -> Remark is invalid (used outside of callback)
    # CHECK: remark_name -> Remark is invalid (used outside of callback)
    # CHECK: category_name -> Remark is invalid (used outside of callback)
    # CHECK: full_category_name -> Remark is invalid (used outside of callback)
    # CHECK: function_name -> Remark is invalid (used outside of callback)
    # CHECK: location -> Remark is invalid (used outside of callback)
    # CHECK: remark_id -> Remark is invalid (used outside of callback)
    # CHECK: args -> Remark is invalid (used outside of callback)
    # CHECK: message -> Remark is invalid (used outside of callback)
    # CHECK: str: <Invalid Remark>
    print("str:", str(remark))


# CHECK-LABEL: TEST: testFilters
@run
def testFilters():
    ctx = Context()
    names = []
    ctx.enable_remarks(
        all_filter="Loop", passed_filter="Vector", callback=collect(names)
    )
    loc = Location.unknown(context=ctx)
    # CHECK: loop passed: True
    print("loop passed:", emit(loc, RemarkKind.PASSED, "LoopPassed", "Loop"))
    # CHECK: loop missed: True
    print("loop missed:", emit(loc, RemarkKind.MISSED, "LoopMissed", "Loop"))
    # CHECK: vector passed: True
    print("vector passed:", emit(loc, RemarkKind.PASSED, "VectorPassed", "Vector"))
    # CHECK: vector missed: False
    print("vector missed:", emit(loc, RemarkKind.MISSED, "VectorMissed", "Vector"))
    # CHECK: other: False
    print("other:", emit(loc, RemarkKind.ANALYSIS, "Other", "Memory"))
    # CHECK: unknown: False
    print("unknown:", emit(loc, RemarkKind.UNKNOWN, "Unknown", "Loop"))
    ctx.finalize_remarks()
    # CHECK: names: ['LoopPassed', 'LoopMissed', 'VectorPassed']
    print("names:", names)


# CHECK-LABEL: TEST: testFinalPolicy
@run
def testFinalPolicy():
    ctx = Context()
    names = []
    ctx.enable_remarks(
        policy=RemarkPolicy.FINAL, all_filter=".*", callback=collect(names)
    )
    loc = Location.unknown(context=ctx)
    emit(loc, RemarkKind.PASSED, "First")
    emit(loc, RemarkKind.MISSED, "Second")
    emit(loc, RemarkKind.FAILURE, "Third")
    emit(loc, RemarkKind.ANALYSIS, "Fourth")
    # CHECK: before finalize: []
    print("before finalize:", names)
    ctx.finalize_remarks()
    # CHECK: after finalize: ['First', 'Fourth', 'Second', 'Third']
    print("after finalize:", sorted(names))


# CHECK-LABEL: TEST: testYamlFile
@run
def testYamlFile():
    ctx = Context()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "remarks.yaml")
    ctx.enable_remarks(output_file=path, all_filter=".*")
    loc = Location.file("test.mlir", 5, 6, context=ctx)
    # CHECK: emitted: True
    print(
        "emitted:",
        emit(
            loc,
            RemarkKind.MISSED,
            "NotUnrolled",
            sub_category="Inner",
            function_name="f",
            message="trip count too small",
            args=[("tripCount", "4")],
        ),
    )
    ctx.finalize_remarks()
    with open(path) as f:
        content = " ".join(f.read().split())
    os.remove(path)
    os.rmdir(directory)
    for fragment in (
        "--- !Missed",
        "Pass: 'Loop:Inner'",
        "Name: NotUnrolled",
        "DebugLoc: { File: test.mlir, Line: 5, Column: 6 }",
        "Function: f",
        "- Remark: trip count too small",
        "- tripCount: '4'",
    ):
        print(fragment, "->", fragment in content)
    # CHECK: --- !Missed -> True
    # CHECK: Pass: 'Loop:Inner' -> True
    # CHECK: Name: NotUnrolled -> True
    # CHECK: DebugLoc: { File: test.mlir, Line: 5, Column: 6 } -> True
    # CHECK: Function: f -> True
    # CHECK: - Remark: trip count too small -> True
    # CHECK: - tripCount: '4' -> True


# CHECK-LABEL: TEST: testBitstreamFile
@run
def testBitstreamFile():
    ctx = Context()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "remarks.bitstream")
    ctx.enable_remarks(
        policy=RemarkPolicy.FINAL,
        output_file=path,
        format=RemarkFormat.BITSTREAM,
        all_filter=".*",
    )
    # CHECK: emitted: True
    print("emitted:", emit(Location.unknown(context=ctx)))
    ctx.finalize_remarks()
    # The file holds an LLVM remark bitstream once finalized.
    with open(path, "rb") as f:
        magic = f.read(4)
    os.remove(path)
    os.rmdir(directory)
    # CHECK: magic: b'RMRK'
    print("magic:", magic)


# CHECK-LABEL: TEST: testEmitAsDiagnostics
@run
def testEmitAsDiagnostics():
    ctx = Context()
    seen = []

    def handler(d):
        seen.append((d.severity, str(d.message)))
        return True

    handle = ctx.attach_diagnostic_handler(handler)
    loc = Location.unknown(context=ctx)
    # Without a sink of its own, the engine emits MLIR remark diagnostics.
    ctx.enable_remarks(all_filter=".*")
    emit(loc, RemarkKind.ANALYSIS, "TripCount", message="trip count is 4")
    ctx.finalize_remarks()
    # CHECK: diagnostics: [(DiagnosticSeverity.REMARK, '[Analysis] TripCount | Category:Loop | Remark="trip count is 4", RemarkId=1')]
    print("diagnostics:", seen)
    seen.clear()
    ctx.enable_remarks(all_filter=".*", print_as_emit_remarks=False)
    # CHECK: silent emitted: True []
    print("silent emitted:", emit(loc), seen)
    ctx.finalize_remarks()
    # A callback is silent by default; combined with the diagnostics it runs first.
    ctx.enable_remarks(
        all_filter=".*",
        callback=lambda r: seen.append("callback"),
        print_as_emit_remarks=True,
    )
    emit(loc)
    ctx.finalize_remarks()
    handle.detach()
    # CHECK: both: ['callback', (DiagnosticSeverity.REMARK, '[Passed] Unroll | Category:Loop | RemarkId=1')]
    print("both:", seen)


# CHECK-LABEL: TEST: testErrors
@run
def testErrors():
    ctx = Context()
    try:
        ctx.enable_remarks(output_file="remarks.yaml", callback=lambda r: None)
        print("no error (unexpected)")
    except ValueError as e:
        # CHECK: ValueError: a remark callback cannot be combined with an output file
        print("ValueError:", e)
    try:
        ctx.enable_remarks(output_file="/nonexistent-dir/remarks.yaml")
        print("no error (unexpected)")
    except ValueError as e:
        # CHECK: ValueError: failed to enable remarks: cannot write '/nonexistent-dir/remarks.yaml'
        print("ValueError:", e)
    # CHECK: enabled after errors: False
    print("enabled after errors:", ctx.remarks_enabled)
    # The options are keyword-only and typed.
    for kwargs in (dict(policy="final"), dict(format="yaml")):
        try:
            ctx.enable_remarks(**kwargs)
            print("no error (unexpected)")
        except TypeError as e:
            print("TypeError:", str(e).splitlines()[0])
    # CHECK-COUNT-2: TypeError: enable_remarks(): incompatible function arguments
    ctx.enable_remarks(all_filter=".*")
    try:
        ctx.enable_remarks(all_filter=".*")
        print("no error (unexpected)")
    except ValueError as e:
        # CHECK: ValueError: remarks are already enabled on this context; call finalize_remarks() first
        print("ValueError:", e)
    # CHECK: still enabled: True
    print("still enabled:", ctx.remarks_enabled)
    loc = Location.unknown(context=ctx)
    try:
        loc.emit_remark("passed", "Name")
        print("no error (unexpected)")
    except TypeError as e:
        # CHECK: TypeError: emit_remark(): incompatible function arguments
        print("TypeError:", str(e).splitlines()[0])
    try:
        Remark()
        print("no error (unexpected)")
    except TypeError as e:
        # CHECK: TypeError: {{.*}}Remark: no constructor defined!
        print("TypeError:", e)
    ctx.finalize_remarks()


# CHECK-LABEL: TEST: testCallbackException
@run
def testCallbackException():
    ctx = Context()
    names = []

    def callback(remark):
        names.append(remark.remark_name)
        if remark.remark_name == "Boom":
            raise RuntimeError("boom")

    ctx.enable_remarks(all_filter=".*", callback=callback)
    loc = Location.unknown(context=ctx)
    # The exception is reported on stderr and dropped.
    # CHECK: boom emitted: True enabled: True
    print("boom emitted:", emit(loc, name="Boom"), "enabled:", ctx.remarks_enabled)
    # CHECK: after emitted: True
    print("after emitted:", emit(loc, name="After"))
    # CHECK: names: ['Boom', 'After']
    print("names:", names)
    ctx.finalize_remarks()


# CHECK-LABEL: TEST: testMultipleContexts
@run
def testMultipleContexts():
    ctx1, ctx2 = Context(), Context()
    seen1, seen2 = [], []
    ctx1.enable_remarks(all_filter=".*", callback=collect(seen1))
    ctx2.enable_remarks(passed_filter="Loop", callback=collect(seen2))
    loc1 = Location.unknown(context=ctx1)
    loc2 = Location.unknown(context=ctx2)
    emit(loc1, RemarkKind.PASSED, "A")
    emit(loc2, RemarkKind.PASSED, "B")
    emit(loc2, RemarkKind.MISSED, "C")
    ctx1.finalize_remarks()
    # CHECK: enabled: False True
    print("enabled:", ctx1.remarks_enabled, ctx2.remarks_enabled)
    emit(loc2, RemarkKind.PASSED, "D")
    ctx2.finalize_remarks()
    # CHECK: seen1: ['A']
    # CHECK: seen2: ['B', 'D']
    print("seen1:", seen1)
    print("seen2:", seen2)


# CHECK-LABEL: TEST: testContextDestroyedWithEngine
@run
def testContextDestroyedWithEngine():
    ctx = Context()
    names = []
    ctx.enable_remarks(
        policy=RemarkPolicy.FINAL, all_filter=".*", callback=collect(names)
    )
    emit(Location.unknown(context=ctx), RemarkKind.PASSED, "Pending")
    # A remark postponed until the context's destruction is dropped.
    ctx = None
    gc.collect()
    # CHECK: dropped: []
    print("dropped:", names)
