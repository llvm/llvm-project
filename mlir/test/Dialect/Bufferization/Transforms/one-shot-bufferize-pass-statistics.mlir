// REQUIRES: asserts
// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops" -mlir-pass-statistics 2>&1 | FileCheck %s

// CHECK: OneShotBufferize
// CHECK:  (S) 1 num-buffer-alloc
// CHECK:  (S) 1 num-tensor-in-place
// CHECK:  (S) 2 num-tensor-out-of-place
func.func @read_after_write_conflict(%cst : f32, %idx : index, %idx2 : index)
    -> (f32, f32) {
  %t = "test.dummy_op"() : () -> (tensor<10xf32>)
  %write = tensor.insert %cst into %t[%idx2] : tensor<10xf32>
  %read = "test.some_use"(%t) : (tensor<10xf32>) -> (f32)
  %read2 = tensor.extract %write[%idx] : tensor<10xf32>
  return %read, %read2 : f32, f32
}
