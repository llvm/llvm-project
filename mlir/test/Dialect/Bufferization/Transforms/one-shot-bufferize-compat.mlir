// RUN: mlir-opt %s \
// RUN:     -one-shot-bufferize="allow-unknown-ops create-deallocs=0" \
// RUN:     -split-input-file | \
// RUN: FileCheck %s --check-prefix=CHECK-NODEALLOC

// RUN: mlir-opt %s \
// RUN:     -one-shot-bufferize="allow-unknown-ops create-deallocs=0" \
// RUN:     -buffer-deallocation | \
// RUN: FileCheck %s --check-prefix=CHECK-BUFFERDEALLOC

// CHECK-NODEALLOC-LABEL: func @out_of_place_bufferization
// CHECK-BUFFERDEALLOC-LABEL: func @out_of_place_bufferization
func @out_of_place_bufferization(%t1 : tensor<?xf32>) -> (f32, f32) {
  //     CHECK-NODEALLOC: memref.alloc
  //     CHECK-NODEALLOC: memref.copy
  // CHECK-NODEALLOC-NOT: memref.dealloc

  //     CHECK-BUFFERDEALLOC: %[[alloc:.*]] = memref.alloc
  //     CHECK-BUFFERDEALLOC: memref.copy
  //     CHECK-BUFFERDEALLOC: memref.dealloc %[[alloc]]

  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 5 : index

  // This bufferizes out-of-place. An allocation + copy will be inserted.
  %0 = tensor.insert %cst into %t1[%idx] : tensor<?xf32>

  %1 = tensor.extract %t1[%idx] : tensor<?xf32>
  %2 = tensor.extract %0[%idx] : tensor<?xf32>
  return %1, %2 : f32, f32
}
