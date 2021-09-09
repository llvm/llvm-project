// RUN: mlir-opt %s -canonicalize --split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @memcpy_after_cast
func @memcpy_after_cast(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  // CHECK-NOT: memref.cast
  // CHECK: gpu.memcpy
  %0 = memref.cast %arg0 : memref<10xf32> to memref<?xf32>
  %1 = memref.cast %arg1 : memref<10xf32> to memref<?xf32>
  gpu.memcpy %0, %1 : memref<?xf32>, memref<?xf32>
  return
}

// CHECK-LABEL: @memset_after_cast
func @memset_after_cast(%arg0: memref<10xf32>, %arg1: f32) {
  // CHECK-NOT: memref.cast
  // CHECK: gpu.memset
  %0 = memref.cast %arg0 : memref<10xf32> to memref<?xf32>
  gpu.memset %0, %arg1 : memref<?xf32>, f32
  return
}

// -----

// Test case: Folding of memref.dim(gpu.alloc(%size), %idx) -> %size
// CHECK-LABEL: func @gpu_dim_of_alloc(
//  CHECK-SAME:     %[[SIZE:[0-9a-z]+]]: index
//  CHECK-NEXT:   return %[[SIZE]] : index
func @gpu_dim_of_alloc(%size: index) -> index {
  %0 = gpu.alloc(%size) : memref<?xindex>
  %c0 = constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xindex>
  return %1 : index
}
