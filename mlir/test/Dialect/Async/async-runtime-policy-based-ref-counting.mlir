// RUN: mlir-opt %s -async-runtime-policy-based-ref-counting | FileCheck %s

// CHECK-LABEL: @token_await
// CHECK:         %[[TOKEN:.*]]: !async.token
func.func @token_await(%arg0: !async.token) {
  // CHECK: async.runtime.await %[[TOKEN]]
  // CHECK-NOT: async.runtime.drop_ref
  async.runtime.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @group_await
// CHECK:         %[[GROUP:.*]]: !async.group
func.func @group_await(%arg0: !async.group) {
  // CHECK: async.runtime.await %[[GROUP]]
  // CHECK-NOT: async.runtime.drop_ref
  async.runtime.await %arg0 : !async.group
  return
}

// CHECK-LABEL: @add_token_to_group
// CHECK:         %[[GROUP:.*]]: !async.group
// CHECK:         %[[TOKEN:.*]]: !async.token
func.func @add_token_to_group(%arg0: !async.group, %arg1: !async.token) {
  // CHECK: async.runtime.add_to_group %[[TOKEN]], %[[GROUP]]
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i64}
  async.runtime.add_to_group %arg1, %arg0 : !async.token
  return
}

// CHECK-LABEL: @value_load
// CHECK:         %[[VALUE:.*]]: !async.value<f32>
func.func @value_load(%arg0: !async.value<f32>) {
  // CHECK: async.runtime.load %[[VALUE]]
  // CHECK: async.runtime.drop_ref %[[VALUE]] {count = 1 : i64}
  %0 = async.runtime.load %arg0 : !async.value<f32>
  return
}

// CHECK-LABEL: @error_check
// CHECK:         %[[TOKEN:.*]]: !async.token
func.func @error_check(%arg0: !async.token) {
  // CHECK: async.runtime.is_error %[[TOKEN]]
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i64}
  %0 = async.runtime.is_error %arg0 : !async.token
  return
}
