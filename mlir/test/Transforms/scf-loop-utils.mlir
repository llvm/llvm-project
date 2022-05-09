// RUN: mlir-opt -allow-unregistered-dialect -test-scf-for-utils -mlir-disable-threading %s | FileCheck %s

// CHECK-LABEL: @hoist
//  CHECK-SAME: %[[lb:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME: %[[ub:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME: %[[step:[a-zA-Z0-9]*]]: index
func.func @hoist(%lb: index, %ub: index, %step: index) {
  // CHECK: %[[A:.*]] = "fake_read"() : () -> index
  // CHECK: %[[RES:.*]] = scf.for %[[I:.*]] = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[VAL:.*]] = %[[A]]) -> (index)
  // CHECK:   %[[YIELD:.*]] = "fake_compute"(%[[VAL]]) : (index) -> index
  // CHECK:   scf.yield %[[YIELD]] : index
  // CHECK: "fake_write"(%[[RES]]) : (index) -> ()
  scf.for %i = %lb to %ub step %step {
    %0 = "fake_read"() : () -> (index)
    %1 = "fake_compute"(%0) : (index) -> (index)
    "fake_write"(%1) : (index) -> ()
  }
  return
}

// CHECK-LABEL: @hoist2
//  CHECK-SAME: %[[lb:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME: %[[ub:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME: %[[step:[a-zA-Z0-9]*]]: index
//  CHECK-SAME: %[[extra_arg:[a-zA-Z0-9]*]]: f32
func.func @hoist2(%lb: index, %ub: index, %step: index, %extra_arg: f32) -> f32 {
  // CHECK: %[[A:.*]] = "fake_read"() : () -> index
  // CHECK: %[[RES:.*]]:2 = scf.for %[[I:.*]] = %[[lb]] to %[[ub]] step %[[step]] iter_args(%[[VAL0:.*]] = %[[extra_arg]], %[[VAL1:.*]] = %[[A]]) -> (f32, index)
  // CHECK:   %[[YIELD:.*]] = "fake_compute"(%[[VAL1]]) : (index) -> index
  // CHECK:   scf.yield %[[VAL0]], %[[YIELD]] : f32, index
  // CHECK: "fake_write"(%[[RES]]#1) : (index) -> ()
  // CHECK: return %[[RES]]#0 : f32
  %0 = scf.for %i = %lb to %ub step %step iter_args(%iter = %extra_arg) -> (f32) {
    %0 = "fake_read"() : () -> (index)
    %1 = "fake_compute"(%0) : (index) -> (index)
    "fake_write"(%1) : (index) -> ()
    scf.yield %iter: f32
  }
  return %0: f32
}
