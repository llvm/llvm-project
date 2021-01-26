// RUN: mlir-opt %s | FileCheck %s --dump-input=always

// CHECK-LABEL: @create_token
func @create_token() -> !async.token {
  // CHECK: %0 = async.runtime.create : !async.token
  %0 = async.runtime.create : !async.token
  // CHECK: return %0 : !async.token
  return %0 : !async.token
}

// CHECK-LABEL: @create_value
func @create_value() -> !async.value<f32> {
  // CHECK: %0 = async.runtime.create : !async.value<f32>
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: return %0 : !async.value<f32>
  return %0 : !async.value<f32>
}

// CHECK-LABEL: @create_group
func @create_group() -> !async.group {
  // CHECK: %0 = async.runtime.create : !async.group
  %0 = async.runtime.create : !async.group
  // CHECK: return %0 : !async.group
  return %0 : !async.group
}

// CHECK-LABEL: @set_token_available
func @set_token_available(%arg0: !async.token) {
  // CHECK: async.runtime.set_available %arg0 : !async.token
  async.runtime.set_available %arg0 : !async.token
  return
}

// CHECK-LABEL: @set_value_available
func @set_value_available(%arg0: !async.value<f32>) {
  // CHECK: async.runtime.set_available %arg0 : !async.value<f32>
  async.runtime.set_available %arg0 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_token
func @await_token(%arg0: !async.token) {
  // CHECK: async.runtime.await %arg0 : !async.token
  async.runtime.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func @await_value(%arg0: !async.value<f32>) {
  // CHECK: async.runtime.await %arg0 : !async.value<f32>
  async.runtime.await %arg0 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_group
func @await_group(%arg0: !async.group) {
  // CHECK: async.runtime.await %arg0 : !async.group
  async.runtime.await %arg0 : !async.group
  return
}

// CHECK-LABEL: @await_and_resume_token
func @await_and_resume_token(%arg0: !async.token,
                             %arg1: !async.coro.handle) {
  // CHECK: async.runtime.await_and_resume %arg0, %arg1 : !async.token
  async.runtime.await_and_resume %arg0, %arg1 : !async.token
  return
}

// CHECK-LABEL: @await_and_resume_value
func @await_and_resume_value(%arg0: !async.value<f32>,
                             %arg1: !async.coro.handle) {
  // CHECK: async.runtime.await_and_resume %arg0, %arg1 : !async.value<f32>
  async.runtime.await_and_resume %arg0, %arg1 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_and_resume_group
func @await_and_resume_group(%arg0: !async.group,
                             %arg1: !async.coro.handle) {
  // CHECK: async.runtime.await_and_resume %arg0, %arg1 : !async.group
  async.runtime.await_and_resume %arg0, %arg1 : !async.group
  return
}

// CHECK-LABEL: @resume
func @resume(%arg0: !async.coro.handle) {
  // CHECK: async.runtime.resume %arg0
  async.runtime.resume %arg0
  return
}

// CHECK-LABEL: @store
func @store(%arg0: f32, %arg1: !async.value<f32>) {
  // CHECK: async.runtime.store %arg0, %arg1 : !async.value<f32>
  async.runtime.store %arg0, %arg1 : !async.value<f32>
  return
}

// CHECK-LABEL: @load
func @load(%arg0: !async.value<f32>) -> f32 {
  // CHECK: %0 = async.runtime.load %arg0 : !async.value<f32>
  // CHECK: return %0 : f32
  %0 = async.runtime.load %arg0 : !async.value<f32>
  return %0 : f32
}

// CHECK-LABEL: @add_to_group
func @add_to_group(%arg0: !async.token, %arg1: !async.value<f32>,
                   %arg2: !async.group) {
  // CHECK: async.runtime.add_to_group %arg0, %arg2 : !async.token
  async.runtime.add_to_group %arg0, %arg2 : !async.token
  // CHECK: async.runtime.add_to_group %arg1, %arg2 : !async.value<f32>
  async.runtime.add_to_group %arg1, %arg2 : !async.value<f32>
  return
}

// CHECK-LABEL: @add_ref
func @add_ref(%arg0: !async.token) {
  // CHECK: async.runtime.add_ref %arg0 {count = 1 : i32}
  async.runtime.add_ref %arg0 {count = 1 : i32} : !async.token
  return
}

// CHECK-LABEL: @drop_ref
func @drop_ref(%arg0: !async.token) {
  // CHECK: async.runtime.drop_ref %arg0 {count = 1 : i32}
  async.runtime.drop_ref %arg0 {count = 1 : i32} : !async.token
  return
}
