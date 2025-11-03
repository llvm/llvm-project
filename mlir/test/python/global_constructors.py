# UNSUPPORTED: target=aarch64{{.*}}, target=arm64{{.*}}
# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-supports-jit
import gc, sys, os, tempfile
from mlir.ir import *
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.runtime import *


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


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


# Test JIT callback in global constructor
# CHECK-LABEL: TEST: testJITCallbackInGlobalCtor
def testJITCallbackInGlobalCtor():
    init_cnt = 0

    @ctypes.CFUNCTYPE(None)
    def initCallback():
        nonlocal init_cnt
        init_cnt += 1

    with Context():
        module = Module.parse(
            r"""
llvm.mlir.global_ctors ctors = [@ctor], priorities = [0 : i32], data = [#llvm.zero]
llvm.func @ctor() {
  func.call @init_callback() : () -> ()
  llvm.return
}
func.func private @init_callback() attributes { llvm.emit_c_interface }
        """
        )

        # Setup execution engine
        execution_engine = ExecutionEngine(lowerToLLVM(module))

        # Validate initialization hasn't run yet
        assert init_cnt == 0

        # # Register callback
        execution_engine.register_runtime("init_callback", initCallback)

        # # Initialize and verify
        execution_engine.initialize()
        assert init_cnt == 1
        # # Second initialization should be no-op
        execution_engine.initialize()
        assert init_cnt == 1


run(testJITCallbackInGlobalCtor)
