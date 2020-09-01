// RUN: mlir-opt %s -affine-parallel-normalize -split-input-file | FileCheck %s

// Normalize steps to 1 and lower bounds to 0.

// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0 * 3)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0) -> (d0 * 2 + 1)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: func @normalize_parallel()
func @normalize_parallel() {
  %cst = constant 1.0 : f32
  %0 = alloc() : memref<2x4xf32>
  // CHECK: affine.parallel (%[[i0:.*]], %[[j0:.*]]) = (0, 0) to (4, 2)
  affine.parallel (%i, %j) = (0, 1) to (10, 5) step (3, 2) {
    // CHECK: %[[i1:.*]] = affine.apply [[$MAP0]](%[[i0]])
    // CHECK: %[[j1:.*]] = affine.apply [[$MAP1]](%[[j0]])
    // CHECK: affine.parallel (%[[k0:.*]]) = (0) to (%[[j1]] - %[[i1]])
    affine.parallel (%k) = (%i) to (%j) {
      // CHECK: %[[k1:.*]] = affine.apply [[$MAP2]](%[[i1]], %[[k0]])
      // CHECK: affine.store %{{.*}}, %{{.*}}[%[[i1]], %[[k1]]] : memref<2x4xf32>
      affine.store %cst, %0[%i, %k] : memref<2x4xf32>
    }
  }
  return
}
