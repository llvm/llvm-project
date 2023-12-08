// RUN: mlir-opt -lower-affine %s | FileCheck %s

// Regression test checking that the memref dialect is loaded as dependency by
// the lowering pass. We shouldn't fail trying to create memref.load here.

// CHECK-LABEL: @no_memref_op
func.func @no_memref_op(%arg0: memref<f32>) {
  // CHECK: memref.load
  affine.load %arg0[] : memref<f32>
  return
}
