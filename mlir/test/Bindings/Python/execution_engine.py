# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc, sys
from mlir.ir import *
from mlir.passmanager import *
from mlir.execution_engine import *

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

# Verify capsule interop.
# CHECK-LABEL: TEST: testCapsule
def testCapsule():
  with Context():
    module = Module.parse(r"""
llvm.func @none() {
  llvm.return
}
    """)
    execution_engine = ExecutionEngine(module)
    execution_engine_capsule = execution_engine._CAPIPtr
    # CHECK: mlir.execution_engine.ExecutionEngine._CAPIPtr
    log(repr(execution_engine_capsule))
    execution_engine._testing_release()
    execution_engine1 = ExecutionEngine._CAPICreate(execution_engine_capsule)
    # CHECK: _mlir.execution_engine.ExecutionEngine
    log(repr(execution_engine1))

run(testCapsule)

# Test invalid ExecutionEngine creation
# CHECK-LABEL: TEST: testInvalidModule
def testInvalidModule():
  with Context():
    # Builtin function
    module = Module.parse(r"""
    func @foo() { return }
    """)
    # CHECK: Got RuntimeError:  Failure while creating the ExecutionEngine.
    try:
      execution_engine = ExecutionEngine(module)
    except RuntimeError as e:
      log("Got RuntimeError: ", e)

run(testInvalidModule)

def lowerToLLVM(module):
  import mlir.conversions
  pm = PassManager.parse("convert-std-to-llvm")
  pm.run(module)
  return module

# Test simple ExecutionEngine execution
# CHECK-LABEL: TEST: testInvokeVoid
def testInvokeVoid():
  with Context():
    module = Module.parse(r"""
func @void() attributes { llvm.emit_c_interface } {
  return
}
    """)    
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    # Nothing to check other than no exception thrown here.
    execution_engine.invoke("void")

run(testInvokeVoid)


# Test argument passing and result with a simple float addition.
# CHECK-LABEL: TEST: testInvokeFloatAdd
def testInvokeFloatAdd():
  with Context():
    module = Module.parse(r"""
func @add(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
  %add = std.addf %arg0, %arg1 : f32
  return %add : f32
}
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    # Prepare arguments: two input floats and one result.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    arg0 = c_float_p(42.)
    arg1 = c_float_p(2.)
    res = c_float_p(-1.)
    execution_engine.invoke("add", arg0, arg1, res)
    # CHECK: 42.0 + 2.0 = 44.0
    log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]))

run(testInvokeFloatAdd)


# Test callback
# CHECK-LABEL: TEST: testBasicCallback
def testBasicCallback():
  # Define a callback function that takes a float and an integer and returns a float.
  @ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float, ctypes.c_int)
  def callback(a, b):
    return a/2 + b/2

  with Context():
    # The module just forwards to a runtime function known as "some_callback_into_python".
    module = Module.parse(r"""
func @add(%arg0: f32, %arg1: i32) -> f32 attributes { llvm.emit_c_interface } {
  %resf = call @some_callback_into_python(%arg0, %arg1) : (f32, i32) -> (f32)
  return %resf : f32
}
func private @some_callback_into_python(f32, i32) -> f32 attributes { llvm.emit_c_interface }
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    execution_engine.register_runtime("some_callback_into_python", callback)

    # Prepare arguments: two input floats and one result.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    c_int_p = ctypes.c_int * 1
    arg0 = c_float_p(42.)
    arg1 = c_int_p(2)
    res = c_float_p(-1.)
    execution_engine.invoke("add", arg0, arg1, res)
    # CHECK: 42.0 + 2 = 44.0
    log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]*2))

run(testBasicCallback)
