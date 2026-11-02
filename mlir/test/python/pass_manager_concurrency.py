# RUN: %PYTHON %s
"""Tests that PassManager.run detaches from the Python interpreter.

The core test runs two PassManagers, on independent Contexts, in two threads.
Each pipeline contains the native `test-python-rendezvous` pass (defined in
lib/TestConcurrencyPasses.cpp), which blocks inside runOnOperation until both
threads have arrived at a shared native barrier. The barrier can only be
satisfied while both threads execute native pass code at the same time:

* If PassManager.run releases the GIL (or the interpreter is free-threaded),
  both threads enter their pipelines, meet at the barrier, and both runs
  succeed.
* If PassManager.run holds the GIL, whichever thread enters run() first blocks
  natively while still attached to the interpreter, the second thread can
  never start its own run(), and the first thread fails deterministically
  once the barrier times out.

No part of this relies on timing margins: the failure mode is a guaranteed
timeout, and the success mode only requires that a runnable thread is
eventually scheduled, which the barrier waits for.
"""

import threading
import uuid

from mlir.ir import Context, MLIRError, Module
from mlir.passmanager import PassManager
import mlir.dialects.python_test as python_test

python_test.register_rendezvous_pass()


def unique_barrier_id(prefix):
    """Barriers are process-global and one-shot, so give each use a fresh id."""
    return f"{prefix}-{uuid.uuid4().hex}"


def run_rendezvous(barrier_id, timeout_seconds, participants=2, ctx=None):
    """Runs one rendezvous pipeline on the given (default: fresh) Context."""
    with Context() if ctx is None else ctx:
        module = Module.parse("module {}")
        pm = PassManager.parse(
            "builtin.module(test-python-rendezvous{"
            f"barrier-id={barrier_id} "
            f"participants={participants} "
            f"timeout-seconds={timeout_seconds}}})"
        )
        pm.run(module.operation)


# Sanity-check the failure mode first: a lone participant must time out, the
# pipeline must fail, and the pass's diagnostic must be captured and attached
# to the MLIRError. This also exercises the ErrorCapture diagnostic path.
def testSoloRendezvousTimesOut():
    barrier_id = unique_barrier_id("solo")
    try:
        run_rendezvous(barrier_id, timeout_seconds=1)
        raise AssertionError("expected the pipeline to fail")
    except MLIRError as e:
        messages = [d.message for d in e.error_diagnostics]
        assert any(
            f"rendezvous '{barrier_id}' timed out" in m for m in messages
        ), messages


# The core test: two PassManagers on independent Contexts must be able to
# execute native pass code simultaneously.
def testConcurrentPassManagersRendezvous():
    barrier_id = unique_barrier_id("pair")
    errors = [None, None]

    def worker(idx):
        try:
            run_rendezvous(barrier_id, timeout_seconds=20)
        except Exception as e:  # pylint: disable=broad-except
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "worker thread hung"
    if any(errors):
        raise AssertionError(
            "two PassManagers could not execute concurrently; PassManager.run "
            f"is not releasing the GIL: {errors}"
        )


# A user-attached diagnostic handler must work while the calling thread is
# detached from the interpreter inside PassManager.run: the handler trampoline
# re-attaches around everything it does (creating the Diagnostic object,
# invoking the callback, and tearing the object down afterwards). With
# emit_error_diagnostics set, the pipeline failure diagnostic bypasses the
# internal error capture and reaches the handler instead.
def testDiagnosticHandlerDuringDetachedRun():
    seen = []

    def handler(d):
        # Materialize everything, including the notes tuple, so that the
        # diagnostic's teardown after the callback has Python objects to drop.
        seen.append((d.severity, str(d.message), [str(n) for n in d.notes]))
        return True

    with Context() as ctx:
        ctx.emit_error_diagnostics = True
        handle = ctx.attach_diagnostic_handler(handler)
        barrier_id = unique_barrier_id("diag")
        try:
            run_rendezvous(barrier_id, timeout_seconds=1, ctx=ctx)
            raise AssertionError("expected the pipeline to fail")
        except MLIRError as e:
            # The handler consumed the diagnostic, so nothing was captured.
            assert not e.error_diagnostics, e.error_diagnostics
        finally:
            handle.detach()
    assert len(seen) == 1, seen
    assert f"rendezvous '{barrier_id}' timed out" in seen[0][1], seen
    assert not handle.had_error


if __name__ == "__main__":
    testSoloRendezvousTimesOut()
    testConcurrentPassManagersRendezvous()
    testDiagnosticHandlerDuringDetachedRun()
