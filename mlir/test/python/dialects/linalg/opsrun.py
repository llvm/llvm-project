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


matmul_boiler = """
func @main() -> f32 attributes {llvm.emit_c_interface} {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %A = memref.alloc() : memref<4x16xf32>
  %B = memref.alloc() : memref<16x8xf32>
  %C = memref.alloc() : memref<4x8xf32>
  linalg.fill(%v1, %A) : f32, memref<4x16xf32>
  linalg.fill(%v2, %B) : f32, memref<16x8xf32>
  linalg.fill(%v0, %C) : f32, memref<4x8xf32>

  call @matmul_on_buffers(%A, %B, %C) :
    (memref<4x16xf32>, memref<16x8xf32>, memref<4x8xf32>) -> ()

  %c0 = constant 0 : index
  %0 = memref.load %C[%c0, %c0] : memref<4x8xf32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : f32
}
"""

fill_boiler = """
func @main() -> i32 attributes {llvm.emit_c_interface} {
  %O = memref.alloc() : memref<4x16xi32>
  %min = constant -1000.0 : f64
  %max = constant 1000.0 : f64
  %seed = constant 42 : i32

  call @fill_on_buffers(%min, %max, %seed, %O) :
    (f64, f64, i32, memref<4x16xi32>) -> ()

  %c0 = constant 0 : index
  %0 = memref.load %O[%c0, %c0] : memref<4x16xi32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : i32
}
"""


def transform(module, boilerplate):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  mod = Module.parse(
      str(module.operation.regions[0].blocks[0].operations[0].operation) +
      boilerplate)
  pm = PassManager.parse("func(convert-linalg-to-loops, convert-scf-to-std)," +
                         "convert-vector-to-llvm," + "convert-std-to-llvm")
  pm.run(mod)
  return mod


def test_matmul_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((4, 16), f32), MemRefType.get((16, 8), f32),
          MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out])

    execution_engine = ExecutionEngine(transform(module, matmul_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: 32.0


test_matmul_builtin()


def test_matmul_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((4, 16), f32), MemRefType.get((16, 8), f32),
          MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out], emit_generic=True)

    execution_engine = ExecutionEngine(transform(module, matmul_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: 32.0


test_matmul_generic()


def test_fill_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(f64, f64, i32, MemRefType.get((4, 16), i32))
      def fill_on_buffers(min, max, seed, out):
        linalg.fill_rng_2d(min, max, seed, outs=[out])

    execution_engine = ExecutionEngine(transform(module, fill_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: -480


test_fill_builtin()


def test_fill_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(f64, f64, i32, MemRefType.get((4, 16), i32))
      def fill_on_buffers(min, max, seed, out):
        linalg.fill_rng_2d(min, max, seed, outs=[out], emit_generic=True)

    execution_engine = ExecutionEngine(transform(module, fill_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: -480


test_fill_generic()
