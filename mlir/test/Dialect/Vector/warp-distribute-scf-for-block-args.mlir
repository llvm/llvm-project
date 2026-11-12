// RUN: mlir-opt %s --test-vector-warp-distribute=propagate-distribution | FileCheck %s

// Yielding a loop-carried block argument used to crash when sinking scf.for
// out of gpu.warp_execute_on_lane_0.
// CHECK-LABEL: func.func @warp_scf_for_yield_loop_carried_arg
// CHECK-NOT: gpu.warp_execute_on_lane_0
// CHECK: %[[FOR:.*]] = scf.for
// CHECK-SAME: iter_args(%[[ARG:.*]] = %{{.*}})
// CHECK:   scf.yield %[[ARG]]
// CHECK: return %[[FOR]]
func.func @warp_scf_for_yield_loop_carried_arg(%laneid: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = gpu.warp_execute_on_lane_0(%laneid)[32] -> (index) {
    %loopResult =
        scf.for %i = %c0 to %c1 step %c1 iter_args(%loopCarried = %c0) -> (index) {
      scf.yield %loopCarried : index
    }
    gpu.yield %loopResult : index
  }
  return %result : index
}
