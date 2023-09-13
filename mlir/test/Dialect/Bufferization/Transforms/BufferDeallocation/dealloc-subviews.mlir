// RUN: mlir-opt -verify-diagnostics -buffer-deallocation \
// RUN:   --buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=true -split-input-file %s > /dev/null

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file > /dev/null

// CHECK-LABEL: func @subview
func.func @subview(%arg0 : index, %arg1 : index, %arg2 : memref<?x?xf32>) {
  %0 = memref.alloc() : memref<64x4xf32, strided<[4, 1], offset: 0>>
  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
  to memref<?x?xf32, strided<[?, ?], offset: ?>>
  test.copy(%1, %arg2) :
    (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32>)
  return
}

//      CHECK: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: memref.subview
// CHECK-NEXT: test.copy
// CHECK-NEXT: bufferization.dealloc (%[[ALLOC]] :
// CHECK-SAME:   if (%true)
// CHECK-NEXT: return
