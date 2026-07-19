# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python-stable-abi
#
# Verify that traceback-based auto-location is not supported under the stable
# ABI (abi3) and always produces unknown locations.

import gc
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


# CHECK-LABEL: TEST: testAutoLocationIsUnknown
@run
def testAutoLocationIsUnknown():
    with Context() as ctx, loc_tracebacks():
        ctx.allow_unregistered_dialects = True
        op = Operation.create("custom.op1")
        # CHECK: loc(unknown)
        print(op.location)
