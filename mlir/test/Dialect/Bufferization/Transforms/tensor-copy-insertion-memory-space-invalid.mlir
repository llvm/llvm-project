// RUN: mlir-opt %s -test-tensor-copy-insertion="must-infer-memory-space" -split-input-file -verify-diagnostics

// An alloc is inserted but the copy is emitted. Therefore, the memory space
// should be specified on the alloc_tensor op.
func.func @memory_space_of_unknown_op() -> (tensor<10xf32>, tensor<10xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // expected-error @+1 {{could not infer memory space}}
  %t = bufferization.alloc_tensor() : tensor<10xf32>
  %s = tensor.insert %cst into %t[%c0] : tensor<10xf32>
  return %s, %t : tensor<10xf32>, tensor<10xf32>
}
