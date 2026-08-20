// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=128" | FileCheck %s

// Regression tests: vectorize inner loops whose bounds depend on the outer loop
// induction variable. Previously this caused a crash due to a use-after-free
// on the scalar outer loop's block argument when erasing the loop nest.

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 6)>

// Both bounds depend on the outer IV.
// CHECK-LABEL: func @inner_loop_dep_bounds
// CHECK: affine.for %[[i:.*]] = 0 to 6 step 32
// CHECK:   affine.for %[[j:.*]] = #map(%[[i]]) to #map1(%[[i]]) step 128
// CHECK:     vector.transfer_write
func.func @inner_loop_dep_bounds(%arg0: memref<32xf32>) {
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 6 step 32 {
    affine.for %j = #map(%i) to #map1(%i) {
      affine.store %cst, %arg0[%j] : memref<32xf32>
    }
  }
  return
}

// Only upper bound depends on the outer IV; lower bound is a constant.
// CHECK-LABEL: func @inner_loop_dep_upper_bound
// CHECK: affine.for %[[i:.*]] = 0 to 64 step 32
// CHECK:   affine.for %[[j:.*]] = 0 to #map1(%[[i]]) step 128
// CHECK:     vector.transfer_write
func.func @inner_loop_dep_upper_bound(%arg0: memref<64xf32>) {
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 64 step 32 {
    affine.for %j = 0 to #map1(%i) {
      affine.store %cst, %arg0[%j] : memref<64xf32>
    }
  }
  return
}
