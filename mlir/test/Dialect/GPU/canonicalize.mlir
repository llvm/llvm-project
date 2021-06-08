// RUN: mlir-opt %s -canonicalize --split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @memcpy_after_cast
func @memcpy_after_cast(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  // CHECK-NOT: memref.cast
  // CHECK: gpu.memcpy
  %0 = memref.cast %arg0 : memref<10xf32> to memref<?xf32>
  %1 = memref.cast %arg1 : memref<10xf32> to memref<?xf32>
  gpu.memcpy %0,%1 : memref<?xf32>, memref<?xf32>
  return
}
