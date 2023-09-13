// RUN: mlir-opt -one-shot-bufferize="bufferize-function-boundaries" -split-input-file %s -verify-diagnostics

// expected-error @below{{failed to bufferize op}}
// expected-error @below{{incoming operands of block argument have inconsistent memory spaces}}
func.func @inconsistent_memory_space() -> tensor<5xf32> {
  %0 = bufferization.alloc_tensor() {memory_space = 0 : ui64} : tensor<5xf32>
  cf.br ^bb1(%0: tensor<5xf32>)
^bb1(%arg1: tensor<5xf32>):
  func.return %arg1 : tensor<5xf32>
^bb2():
  %1 = bufferization.alloc_tensor() {memory_space = 1 : ui64} : tensor<5xf32>
  cf.br ^bb1(%1: tensor<5xf32>)
}

// -----

// expected-error @below{{failed to bufferize op}}
// expected-error @below{{could not infer buffer type of block argument}}
func.func @cannot_infer_type() {
  return
  // The type of the block argument cannot be inferred.
^bb1(%t: tensor<5xf32>):
  cf.br ^bb1(%t: tensor<5xf32>)
}
