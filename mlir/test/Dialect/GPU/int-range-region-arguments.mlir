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
