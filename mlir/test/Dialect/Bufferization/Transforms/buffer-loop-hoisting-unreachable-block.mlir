// RUN: mlir-opt -buffer-loop-hoisting %s

func.func @unreachable_block() {
  return
^bb1:
  %alloc = memref.alloc() : memref<2xf32>
  memref.dealloc %alloc : memref<2xf32>
  return
}
