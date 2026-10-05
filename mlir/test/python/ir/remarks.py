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


def emit(loc, kind="passed", name="Unroll", category="Loop", **kwargs):
    return loc.emit_remark(kind, name, category=category, **kwargs)


# CHECK-LABEL: TEST: testNoEngine
@run
def testNoEngine():
    ctx = Context()
    with ctx:
        loc = Location.file("test.mlir", 1, 2)
        # CHECK: enabled: False
        print("enabled:", ctx.remarks_enabled)
        # CHECK: emitted: False
        print("emitted:", emit(loc))
        # Finalizing without an engine is a no-op.
        ctx.finalize_remarks()


# CHECK-LABEL: TEST: testCallback
@run
def testCallback():
    ctx = Context()
    collected = []

    def callback(remark):
        collected.append(remark)
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
        )
        print("args:", remark.args)
        print("message:", remark.message)
        print("str:", str(remark))
        print("location:", remark.location)
        print("id:", remark.remark_id)

    ctx.enable_remarks(all_filter=".*", callback=callback)
    # CHECK: enabled: True
    print("enabled:", ctx.remarks_enabled)
    with ctx:
        loc = Location.file("test.mlir", 3, 4)
        # CHECK: kind: RemarkKind.PASSED name: Unroll category: Loop full: Loop:Inner function: main
        # CHECK: args: [('RemarkId', '1'), ('Remark', 'unrolled by 4'), ('factor', '4')]
        # CHECK: message: [Passed] Unroll | Category:Loop:Inner | Function=main | Remark="unrolled by 4", RemarkId=1, factor=4
        # CHECK: str: [Passed] Unroll | Category:Loop:Inner | Function=main | Remark="unrolled by 4", RemarkId=1, factor=4
        # CHECK: location: loc("test.mlir":3:4)
        # CHECK: id: 1
        # CHECK: emitted: True
        emitted = emit(
            loc,
            RemarkKind.PASSED,
            sub_category="Inner",
            function_name="main",
            message="unrolled by 4",
            args=[("factor", "4")],
        )
        print("emitted:", emitted)
    # A remark is invalid once the callback has returned.
    try:
        collected[0].remark_name
    except ValueError as e:
        # CHECK: invalid: Remark is invalid (used outside of callback)
        print("invalid:", e)
    # CHECK: invalid str: <Invalid Remark>
    print("invalid str:", str(collected[0]))
    collected.clear()
    ctx.finalize_remarks()
    # CHECK: enabled after finalize: False
    print("enabled after finalize:", ctx.remarks_enabled)


# CHECK-LABEL: TEST: testFilters
@run
def testFilters():
    ctx = Context()
    names = []
    ctx.enable_remarks(
        all_filter="Loop",
        passed_filter="Vector",
        callback=lambda r: names.append(r.remark_name),
    )
    with ctx:
        loc = Location.unknown()
        # The all-filter admits every kind of the Loop category.
        # CHECK: loop passed: True
        print("loop passed:", emit(loc, "passed", "LoopPassed", "Loop"))
        # CHECK: loop missed: True
        print("loop missed:", emit(loc, "missed", "LoopMissed", "Loop"))
        # The passed-filter admits only passed remarks of the Vector category.
        # CHECK: vector passed: True
        print("vector passed:", emit(loc, "passed", "VectorPassed", "Vector"))
        # CHECK: vector missed: False
        print("vector missed:", emit(loc, "missed", "VectorMissed", "Vector"))
        # CHECK: other: False
        print("other:", emit(loc, "analysis", "Other", "Memory"))
        # Unknown kinds are never emitted.
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
        policy="final", all_filter=".*", callback=lambda r: names.append(r.remark_name)
    )
    with ctx:
        loc = Location.unknown()
        emit(loc, "passed", "First")
        emit(loc, "missed", "Second")
        emit(loc, "failed", "Third")
        emit(loc, "analysis", "Fourth")
    # Nothing is delivered until finalize.
    # CHECK: before finalize: []
    print("before finalize:", names)
    ctx.finalize_remarks()
    # CHECK: after finalize: 4
    print("after finalize:", len(names))
    assert sorted(names) == ["First", "Fourth", "Second", "Third"]
    # The engine can be enabled again after finalize.
    ctx.enable_remarks(all_filter=".*", callback=lambda r: names.append(r.remark_name))
    with ctx:
        emit(Location.unknown(), "passed", "Fifth")
    # CHECK: re-enabled: Fifth
    print("re-enabled:", names[-1])
    ctx.finalize_remarks()


# CHECK-LABEL: TEST: testYamlFile
@run
def testYamlFile():
    ctx = Context()
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "remarks.yaml")
    ctx.enable_remarks(format="yaml", output_file=path, all_filter=".*")
    with ctx:
        loc = Location.file("test.mlir", 5, 6)
        # CHECK: emitted: True
        print("emitted:", emit(loc, "missed", "NotUnrolled", "Loop", function_name="f"))
    ctx.finalize_remarks()
    with open(path) as f:
        content = f.read()
    # CHECK: has file: True
    print("has file:", os.path.exists(path))
    # CHECK: has name: True
    print("has name:", "NotUnrolled" in content)
    # CHECK: has pass: True
    print("has pass:", "Loop" in content)
    os.remove(path)
    os.rmdir(directory)


# CHECK-LABEL: TEST: testEmitFormat
@run
def testEmitFormat():
    ctx = Context()
    messages = []

    def handler(d):
        messages.append((d.severity, str(d.message)))
        return True

    handler_handle = ctx.attach_diagnostic_handler(handler)
    ctx.enable_remarks(all_filter=".*")
    with ctx:
        emit(
            Location.unknown(),
            "analysis",
            "TripCount",
            "Loop",
            message="trip count is 4",
        )
    ctx.finalize_remarks()
    handler_handle.detach()
    # CHECK: diagnostics: [(DiagnosticSeverity.REMARK, '[Analysis] TripCount | Category:Loop | Remark="trip count is 4", RemarkId=1')]
    print("diagnostics:", messages)


# CHECK-LABEL: TEST: testCallbackAndEmit
@run
def testCallbackAndEmit():
    ctx = Context()
    seen = []

    def handler(d):
        seen.append("diagnostic")
        return True

    handler_handle = ctx.attach_diagnostic_handler(handler)
    ctx.enable_remarks(
        all_filter=".*",
        callback=lambda r: seen.append("callback"),
        print_as_emit_remarks=True,
    )
    with ctx:
        emit(Location.unknown())
    ctx.finalize_remarks()
    handler_handle.detach()
    # CHECK: seen: ['callback', 'diagnostic']
    print("seen:", seen)


# CHECK-LABEL: TEST: testErrors
@run
def testErrors():
    ctx = Context()
    for kwargs in (
        dict(policy="sometimes"),
        dict(format="xml"),
        dict(format="yaml", callback=lambda r: None),
        dict(format="yaml", output_file="/nonexistent-dir/remarks.yaml"),
    ):
        try:
            ctx.enable_remarks(all_filter=".*", **kwargs)
            print("no error (unexpected)")
        except ValueError as e:
            print("ValueError:", e)
    # CHECK: ValueError: unknown remark policy 'sometimes'; expected 'all' or 'final'
    # CHECK: ValueError: unknown remark format 'xml'; expected 'emit', 'yaml' or 'bitstream'
    # CHECK: ValueError: a remark callback cannot be combined with format='yaml'; use format='emit'
    # CHECK: ValueError: failed to enable remarks: cannot write '/nonexistent-dir/remarks.yaml'
    ctx.enable_remarks(all_filter=".*")
    try:
        ctx.enable_remarks(all_filter=".*")
    except ValueError as e:
        # CHECK: ValueError: remarks are already enabled on this context; call finalize_remarks() first
        print("ValueError:", e)
    with ctx:
        loc = Location.unknown()
        try:
            loc.emit_remark("sideways", "Name")
        except ValueError as e:
            # CHECK: ValueError: unknown remark kind 'sideways'; expected 'passed', 'missed', 'failure' or 'analysis'
            print("ValueError:", e)
        try:
            loc.emit_remark(3, "Name")
        except TypeError as e:
            # CHECK: TypeError: remark kind must be a RemarkKind or a string
            print("TypeError:", e)
    ctx.finalize_remarks()


# CHECK-LABEL: TEST: testCallbackException
@run
def testCallbackException():
    ctx = Context()

    def callback(remark):
        raise RuntimeError("boom")

    ctx.enable_remarks(all_filter=".*", callback=callback)
    with ctx:
        # The exception is printed to stderr and dropped; emission still succeeds.
        # CHECK: emitted: True
        print("emitted:", emit(Location.unknown()))
    ctx.finalize_remarks()


# CHECK-LABEL: TEST: testContextDestroyedWithEngine
@run
def testContextDestroyedWithEngine():
    ctx = Context()
    names = []
    ctx.enable_remarks(
        policy="final", all_filter=".*", callback=lambda r: names.append(r.remark_name)
    )
    with ctx:
        emit(Location.unknown(), "passed", "Pending")
    # Destroying the context finalizes the engine; a postponed remark cannot
    # reach Python anymore and is dropped, without crashing.
    ctx = None
    gc.collect()
    # CHECK: dropped: []
    print("dropped:", names)
