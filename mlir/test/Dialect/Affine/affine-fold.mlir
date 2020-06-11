// RUN: mlir-opt -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @affine_parallel_rank0
func @affine_parallel_rank0() {
  // CHECK-NEXT: constant
  %cst = constant 1.0 : f32
  // CHECK-NEXT: alloc
  %0 = alloc() : memref<f32>
  // CHECK-NEXT: affine.store
  affine.parallel () = () to () {
    affine.store %cst, %0[] : memref<f32>
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @affine_parallel_range1
func @affine_parallel_range1() {
  // CHECK-NEXT: constant
  %cst = constant 1.0 : f32
  // CHECK-NEXT: alloc
  %0 = alloc() : memref<2x4xf32>
  // CHECK-NEXT: affine.store
  affine.parallel (%i, %j) = (0, 1) to (1, 2) {
    affine.store %cst, %0[%i, %j] : memref<2x4xf32>
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @affine_parallel_partial_range1
func @affine_parallel_partial_range1() {
  // CHECK-NEXT: constant
  %cst = constant 1.0 : f32
  // CHECK-NEXT: alloc
  %0 = alloc() : memref<2x4xf32>
  // CHECK-NEXT: affine.parallel (%{{.*}}) = (0) to (10)
  affine.parallel (%i, %j) = (0, 1) to (10, 2) {
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%{{.*}}, 1]
    affine.store %cst, %0[%i, %j] : memref<2x4xf32>
  }
  // CHECK: return
  return
}
