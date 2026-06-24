# RUN: %PYTHON %s | FileCheck %s

"""Tests for MBarrierTestWaitOp and MBarrierTryWaitOp Python bindings.
Covers the none-phase (single-result i1) variant.
Two construction styles: OpView class and free function.
"""

from mlir.ir import *
from mlir.dialects import nvvm, func


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: TEST: test_mbarrier_wait
@run
def test_mbarrier_wait():
    """MBarrierTestWaitOp and MBarrierTryWaitOp -- OpView and free-function styles."""
    i64 = IntegerType.get_signless(64)
    ptr = Type.parse("!llvm.ptr<3>")

    @func.FuncOp.from_py_func(ptr, i64)
    def none_phase(addr, state):
        op_test = nvvm.MBarrierTestWaitOp(addr=addr, stateOrPhase=state)
        assert op_test.res is not None
        op_try = nvvm.MBarrierTryWaitOp(addr=addr, stateOrPhase=state)
        assert op_try.res is not None
        wc_test = nvvm.mbarrier_test_wait(addr=addr, state_or_phase=state)
        assert wc_test is not None
        wc_try = nvvm.mbarrier_try_wait(addr=addr, state_or_phase=state)
        assert wc_try is not None


# CHECK: func.func @none_phase
# CHECK:   nvvm.mbarrier.test.wait %{{.*}}, %{{.*}} : !llvm.ptr<3>, i64 -> i1
# CHECK:   nvvm.mbarrier.try_wait %{{.*}}, %{{.*}} : !llvm.ptr<3>, i64 -> i1
# CHECK:   nvvm.mbarrier.test.wait %{{.*}}, %{{.*}} : !llvm.ptr<3>, i64 -> i1
# CHECK:   nvvm.mbarrier.try_wait %{{.*}}, %{{.*}} : !llvm.ptr<3>, i64 -> i1
