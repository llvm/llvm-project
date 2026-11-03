// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

func.func private @consume(index)

// CHECK-LABEL: func @delayed_launch_dimension
// CHECK: gpu.launch
// CHECK: test.reflect_bounds <umin = 0 : index, umax = 1 : index, smin = 0 : index, smax = 1 : index>
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[MERGE:.*]] = scf.if
// CHECK: scf.yield %[[SELECT]]
// CHECK: call @consume(%[[MERGE]])
func.func @delayed_launch_dimension(%c: i1, %d: i1) {
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %zero = arith.constant 0 : index
  %dim = scf.execute_region -> index {
    scf.yield %two : index
  }
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
      threads(%tx, %ty, %tz) in (%sx = %dim, %sy = %one, %sz = %one) {
    %reflected = test.reflect_bounds %tx : index
    %selected = arith.select %c, %tx, %zero : index
    %merged = scf.if %d -> index {
      scf.yield %selected : index
    } else {
      scf.yield %zero : index
    }
    func.call @consume(%merged) : (index) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @cluster_arguments
// CHECK: gpu.launch clusters
// CHECK-COUNT-3: test.reflect_bounds <umin = 0 : index, umax = 4294967294 : index, smin = 0 : index, smax = 4294967294 : index>
// CHECK-COUNT-3: test.reflect_bounds <umin = 1 : index, umax = 4294967295 : index, smin = 1 : index, smax = 4294967295 : index>
// CHECK: %[[SELECT:.*]] = arith.select
// CHECK: %[[MERGE:.*]] = scf.if
// CHECK: scf.yield %[[SELECT]]
// CHECK: memref.store %[[MERGE]]
func.func @cluster_arguments(%c: i1, %d: i1, %out: memref<1xindex>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  gpu.launch clusters(%cx, %cy, %cz) in (%sx = %one, %sy = %one, %sz = %one)
      blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
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
