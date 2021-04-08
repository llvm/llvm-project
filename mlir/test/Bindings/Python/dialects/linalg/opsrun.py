# RUN: %PYTHON %s 2>&1 | FileCheck %s

import sys
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()

boilerplate = """
func @main() -> f32 attributes {llvm.emit_c_interface} {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %A = memref.alloc() : memref<4x16xf32>
  %B = memref.alloc() : memref<16x8xf32>
  %C = memref.alloc() : memref<4x8xf32>
  linalg.fill(%A, %v1) : memref<4x16xf32>, f32
  linalg.fill(%B, %v2) : memref<16x8xf32>, f32
  linalg.fill(%C, %v0) : memref<4x8xf32>, f32

  call @matmul_on_buffers(%A, %B, %C) : 
    (memref<4x16xf32>, memref<16x8xf32>, memref<4x8xf32>) -> ()

  %c0 = constant 0 : index
  %0 = memref.load %C[%c0, %c0] : memref<4x8xf32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : f32
}
"""

def transform(module):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  mod = Module.parse(
    str(module.operation.regions[0].blocks[0].operations[0].operation) +
    boilerplate)
  pm = PassManager.parse("func(convert-linalg-to-loops, convert-scf-to-std)," + 
                         "convert-vector-to-llvm," + 
                         "convert-std-to-llvm")
  pm.run(mod)
  return mod

def test_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(MemRefType.get((4, 16), f32),
                                   MemRefType.get((16, 8), f32),
                                   MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out])
    
    execution_engine = ExecutionEngine(transform(module))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke("main", res)

    log('RESULT: ', res[0])
    # CHECK: RESULT: 32.0

test_builtin()

def test_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(MemRefType.get((4, 16), f32),
                                   MemRefType.get((16, 8), f32),
                                   MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out], emit_generic=True)
    
    execution_engine = ExecutionEngine(transform(module))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke("main", res)

    log('RESULT: ', res[0])
    # CHECK: RESULT: 32.0

test_generic()
