// RUN: mlir-opt %s --loop-invariant-code-motion | FileCheck %s

func.func private @side_effect(%arg0 : f32, %arg1 : f32, %arg2 : f32)

// CHECK-LABEL: func @broadcast_hoisting
//  CHECK-SAME: (%[[ARG:.*]]: f32, %[[IDX:.*]]: i32)
func.func @broadcast_hoisting(%arg0 : f32, %arg1 : i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
// CHECK: %[[V1:.*]] = gpu.broadcast_lane %[[ARG]], any_lane : f32
// CHECK: %[[V2:.*]] = gpu.broadcast_lane %[[ARG]], lane %[[IDX]] : f32
// CHECK: scf.for
// CHECK: %[[V0:.*]] = gpu.broadcast_lane %[[ARG]], first_lane : f32
// CHECK: func.call @side_effect(%[[V0]], %[[V1]], %[[V2]])
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = gpu.broadcast_lane %arg0, first_lane : f32
    %1 = gpu.broadcast_lane %arg0, any_lane : f32
    %2 = gpu.broadcast_lane %arg0, lane %arg1 : f32
    func.call @side_effect(%0, %1, %2) : (f32, f32, f32) -> ()
  }
  func.return
}
