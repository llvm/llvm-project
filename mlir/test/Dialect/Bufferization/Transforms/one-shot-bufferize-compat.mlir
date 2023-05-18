// RUN: mlir-opt %s \
// RUN:     -one-shot-bufferize="allow-unknown-ops" \
// RUN:     -split-input-file | \
// RUN: FileCheck %s

// CHECK-LABEL: func @out_of_place_bufferization
func.func @out_of_place_bufferization(%t1 : tensor<?xf32>) -> (f32, f32) {
  //     CHECK: memref.alloc
  //     CHECK: memref.copy
  // CHECK-NOT: memref.dealloc

  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 5 : index

  // This bufferizes out-of-place. An allocation + copy will be inserted.
  %0 = tensor.insert %cst into %t1[%idx] : tensor<?xf32>

  %1 = tensor.extract %t1[%idx] : tensor<?xf32>
  %2 = tensor.extract %0[%idx] : tensor<?xf32>
  return %1, %2 : f32, f32
}
