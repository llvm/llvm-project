# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python-stable-abi
#
# Verify that traceback-based locations work under the stable ABI (abi3).
# The limited API path uses sys._getframe() and cannot provide column info,
# so locations have :0 for columns. This test checks that function names,
# file paths, callsite nesting, and frame limiting all work correctly.

import gc
from mlir.ir import *
from mlir.dialects import arith


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testStableABILocations
@run
def testStableABILocations():
    with Context() as ctx, loc_tracebacks():
        ctx.allow_unregistered_dialects = True

        # Basic op creation produces a callsite location with function names.
        op = Operation.create("custom.op1")
        # CHECK: loc(callsite("testStableABILocations"({{.*}}auto_location_stable_abi.py":28:0)
        # CHECK-SAME: at callsite("run"({{.*}}auto_location_stable_abi.py":16:0)
        # CHECK-SAME: at "<module>"({{.*}}auto_location_stable_abi.py":22:0))))
        print(op.location)

        # Nested function calls produce nested callsite locations.
        def inner():
            return arith.constant(IndexType.get(), 1)

        val = inner()
        # CHECK: loc(callsite(
        # CHECK-SAME: "testStableABILocations.<locals>.inner"({{.*}}auto_location_stable_abi.py":36:0)
        # CHECK-SAME: at callsite("testStableABILocations"({{.*}}auto_location_stable_abi.py":38:0)
        # CHECK-SAME: at callsite("run"({{.*}}auto_location_stable_abi.py":16:0)
        # CHECK-SAME: at "<module>"({{.*}}auto_location_stable_abi.py":22:0)))))))
        print(val.location)

        # Frame limit of 0 produces unknown location.
        from mlir.dialects._ods_common import _cext

        _cext.globals.set_loc_tracebacks_frame_limit(0)
        val2 = arith.constant(IndexType.get(), 2)
        # CHECK: loc(unknown)
        print(val2.location)
