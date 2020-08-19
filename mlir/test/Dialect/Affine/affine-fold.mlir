// RUN: mlir-opt -canonicalize -split-input-file %s | FileCheck %s

// CHECK: func @affine_parallel_rank0
func @affine_parallel_rank0(%out: memref<f32>) {
  // CHECK-NEXT: constant
  %cst = constant 0.0 : f32
  // CHECK-NEXT: affine.store
  affine.parallel () = () to () {
    affine.parallel () = () to () {
      affine.store %cst, %out[] : memref<f32>
    }
  }
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
  affine.parallel (%i, %j) = (0, 1) to (2, 2) step (2, 1) {
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

// -----

// CHECK-LABEL: func @simplify_parallel
func @simplify_parallel() {
  %cst = constant 1.0 : f32
  %0 = alloc() : memref<2x4xf32>
  // CHECK: affine.parallel (%[[i:.*]], %[[j:.*]]) = (0, 0) to (4, 2) {
  affine.parallel (%i, %j) = (0, 1) to (10, 5) step (3, 2) {
    // CHECK: affine.parallel (%[[k:.*]]) = (0) to (%[[j]] * 2 - %[[i]] * 3 + 1) {
    affine.parallel (%k) = (%i) to (%j) {
      // CHECK: affine.store %{{.*}}, %{{.*}}[%[[i]] * 3, %[[i]] * 3 + %[[k]]] : memref<2x4xf32>
      affine.store %cst, %0[%i, %k] : memref<2x4xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @affine_parallel_const_bounds
func @affine_parallel_const_bounds() {
  %cst = constant 1.0 : f32
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %0 = alloc() : memref<4xf32>
  // CHECK: affine.parallel (%{{.*}}) = (0) to (4)
  affine.parallel (%i) = (%c0) to (%c0 + %c4) {
    affine.store %cst, %0[%i] : memref<4xf32>
  }
  return
}

// -----

#map0 = affine_map<(d0) -> (d0 * 5)>
#map1 = affine_map<(d0) -> (d0 * 10)>

// CHECK-LABEL: func @affine_parallel_fold_bounds
func @affine_parallel_fold_bounds() {
  %cst = constant 1.0 : f32
  %0 = alloc() : memref<100x100xf32>
  // CHECK: affine.parallel (%[[i0:.*]], %[[j0:.*]]) =
  affine.parallel (%i0, %j0) = (0, 0) to (100, 10) {
    %2 = affine.apply #map0(%i0)
    %3 = affine.apply #map1(%j0)
    // CHECK-NOT: affine.apply
    // CHECK: affine.parallel (%[[i1:.*]], %[[j1:.*]]) = (0, 0) to (5, 10) {
    affine.parallel (%i1, %j1) = (%2, %3) to (%2 + 5, %3 + 10) {
      // CHECK: affine.store %{{.*}}, %{{.*}}[%[[i0]] * 5 + %[[i1]], %[[j0]] * 10 + %[[j1]]]
      affine.store %cst, %0[%i1, %j1] : memref<100x100xf32>
    }
  }
  return
}
