// RUN: mlir-opt %s --loop-invariant-code-motion | FileCheck %s

func.func private @side_effect(%arg0 : f32, %arg1 : f32)
func.func private @use_i32(%arg0 : i32)

// CHECK-LABEL: func @broadcast_hoisting
//  CHECK-SAME: (%[[ARG:.*]]: f32, %[[IDX:.*]]: i32, {{.*}}: index)
func.func @broadcast_hoisting(%arg0 : f32, %arg1 : i32, %arg2 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
// `specific_lane` can be speculated across the control flow, but
// `first_active_lane` cannot as active lanes can change.
// CHECK: %[[V1:.*]] = gpu.subgroup_broadcast %[[ARG]], specific_lane %[[IDX]] : f32
// CHECK: scf.for
// CHECK: %[[V0:.*]] = gpu.subgroup_broadcast %[[ARG]], first_active_lane : f32
// CHECK: func.call @side_effect(%[[V0]], %[[V1]])
  scf.for %i = %c0 to %arg2 step %c1 {
    %0 = gpu.subgroup_broadcast %arg0, first_active_lane : f32
    %1 = gpu.subgroup_broadcast %arg0, specific_lane %arg1 : f32
    func.call @side_effect(%0, %1) : (f32, f32) -> ()
  }
  func.return
}

// CHECK-LABEL: func @ballot_no_hoisting
//  CHECK-SAME: (%[[PRED:.*]]: i1, {{.*}}: index)
func.func @ballot_no_hoisting(%pred: i1, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Ballot cannot be speculated across control flow because
  // it depends on active lanes, which can change.
  // CHECK: scf.for
  // CHECK: %[[BALLOT:.*]] = gpu.ballot %[[PRED]] : i32
  // CHECK: func.call @use_i32(%[[BALLOT]])
  scf.for %i = %c0 to %n step %c1 {
    %0 = gpu.ballot %pred : i32
    func.call @use_i32(%0) : (i32) -> ()
  }
  func.return
}
