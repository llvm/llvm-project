// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

// CHECK-LABEL: func @cluster_arguments
// CHECK: gpu.launch clusters
// CHECK-COUNT-3: test.reflect_bounds <umin = 0 : index, umax = 4294967294 : index, smin = 0 : index, smax = 4294967294 : index>
// CHECK-COUNT-3: test.reflect_bounds <umin = 1 : index, umax = 4294967295 : index, smin = 1 : index, smax = 4294967295 : index>
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[MERGE:.*]] = scf.if
// CHECK: scf.yield %[[SELECT]]
// CHECK: memref.store %[[MERGE]]
func.func @cluster_arguments(%c: i1, %d: i1, %size: index, %out: memref<1xindex>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  gpu.launch clusters(%cx, %cy, %cz) in (%sx = %one, %sy = %one, %sz = %one)
      blocks(%bx, %by, %bz) in (%gx = %size, %gy = %size, %gz = %size)
      threads(%tx, %ty, %tz) in (%dx = %one, %dy = %one, %dz = %one) {
    %cx_ref = test.reflect_bounds %cx : index
    %cy_ref = test.reflect_bounds %cy : index
    %cz_ref = test.reflect_bounds %cz : index
    %sx_ref = test.reflect_bounds %sx : index
    %sy_ref = test.reflect_bounds %sy : index
    %sz_ref = test.reflect_bounds %sz : index
    %selected = arith.select %c, %sx, %zero : index
    %merged = scf.if %d -> index {
      scf.yield %selected : index
    } else {
      scf.yield %zero : index
    }
    memref.store %merged, %out[%zero] : memref<1xindex>
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @cluster_input_ranges
func.func @cluster_input_ranges() {
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %twelve = arith.constant 12 : index
  %grid = test.with_bounds <umin = 12 : index, umax = 24 : index,
                           smin = 12 : index, smax = 24 : index> : index
  %cluster = test.with_bounds <umin = 2 : index, umax = 4 : index,
                              smin = 2 : index, smax = 4 : index> : index
  gpu.launch clusters(%cx, %cy, %cz) in (%sx = %three, %sy = %cluster, %sz = %one)
      blocks(%bx, %by, %bz) in (%gx = %twelve, %gy = %grid, %gz = %grid)
      threads(%tx, %ty, %tz) in (%dx = %one, %dy = %one, %dz = %one) {
    // CHECK: test.reflect_bounds <umin = 0 : index, umax = 3 : index, smin = 0 : index, smax = 3 : index>
    %cx_ref = test.reflect_bounds %cx : index
    // CHECK: test.reflect_bounds <umin = 0 : index, umax = 11 : index, smin = 0 : index, smax = 11 : index>
    %cy_ref = test.reflect_bounds %cy : index
    // CHECK: test.reflect_bounds <umin = 0 : index, umax = 23 : index, smin = 0 : index, smax = 23 : index>
    %cz_ref = test.reflect_bounds %cz : index
    // CHECK: test.reflect_bounds <umin = 4 : index, umax = 4 : index, smin = 4 : index, smax = 4 : index>
    %sx_ref = test.reflect_bounds %sx : index
    // CHECK: test.reflect_bounds <umin = 3 : index, umax = 12 : index, smin = 3 : index, smax = 12 : index>
    %sy_ref = test.reflect_bounds %sy : index
    // CHECK: test.reflect_bounds <umin = 12 : index, umax = 24 : index, smin = 12 : index, smax = 24 : index>
    %sz_ref = test.reflect_bounds %sz : index
    gpu.terminator
  }
  return
}
