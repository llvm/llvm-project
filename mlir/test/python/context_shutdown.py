# RUN: %PYTHON %s
# Regression test: entering a Context (or Location/InsertionPoint) without
# exiting before interpreter shutdown used to segfault. The thread-local
# context stack holds nb::object references; if not cleared before
# Py_Finalize(), the thread_local destructor calls Py_DECREF through the
# dead runtime.

from mlir.ir import *

# Case 1: Single context entered, not exited.
ctx = Context()
ctx.__enter__()
ctx.enable_multithreading(False)
with Location.unknown():
    m = Module.parse("func.func @f() { return }")

# Case 2: Multiple contexts entered, not exited.
ctx2 = Context()
ctx2.__enter__()

# Case 3: Location entered, not exited (also uses the same thread-local stack).
loc = Location.unknown()
loc.__enter__()

# Interpreter shutdown proceeds with contexts/locations still on the
# thread-local stack. Before the fix, this would segfault (exit code 139).
