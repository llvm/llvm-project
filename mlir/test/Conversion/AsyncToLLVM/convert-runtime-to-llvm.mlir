// RUN: mlir-opt %s -convert-async-to-llvm | FileCheck %s

// CHECK-LABEL: @create_token
func @create_token() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  return
}

// CHECK-LABEL: @create_value
func @create_value() {
  // CHECK: %[[NULL:.*]] = llvm.mlir.null : !llvm.ptr<f32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OFFSET:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[OFFSET]]
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue(%[[SIZE]])
  %0 = async.runtime.create : !async.value<f32>
  return
}

// CHECK-LABEL: @create_group
func @create_group() {
  // CHECK: %[[GROUP:.*]] = call @mlirAsyncRuntimeCreateGroup
  %0 = async.runtime.create : !async.group
  return
}

// CHECK-LABEL: @set_token_available
func @set_token_available() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[TOKEN]])
  async.runtime.set_available %0 : !async.token
  return
}

// CHECK-LABEL: @set_value_available
func @set_value_available() {
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: call @mlirAsyncRuntimeEmplaceValue(%[[VALUE]])
  async.runtime.set_available %0 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_token
func @await_token() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: call @mlirAsyncRuntimeAwaitToken(%[[TOKEN]])
  async.runtime.await %0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func @await_value() {
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: call @mlirAsyncRuntimeAwaitValue(%[[VALUE]])
  async.runtime.await %0 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_group
func @await_group() {
  // CHECK: %[[GROUP:.*]] = call @mlirAsyncRuntimeCreateGroup
  %0 = async.runtime.create : !async.group
  // CHECK: call @mlirAsyncRuntimeAwaitAllInGroup(%[[GROUP]])
  async.runtime.await %0 : !async.group
  return
}

// CHECK-LABEL: @await_and_resume_token
func @await_and_resume_token() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %2 = async.runtime.create : !async.token
  // CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
  // CHECK: call @mlirAsyncRuntimeAwaitTokenAndExecute
  // CHECK-SAME: (%[[TOKEN]], %[[HDL]], %[[RESUME]])
  async.runtime.await_and_resume %2, %1 : !async.token
  return
}

// CHECK-LABEL: @await_and_resume_value
func @await_and_resume_value() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %2 = async.runtime.create : !async.value<f32>
  // CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
  // CHECK: call @mlirAsyncRuntimeAwaitValueAndExecute
  // CHECK-SAME: (%[[VALUE]], %[[HDL]], %[[RESUME]])
  async.runtime.await_and_resume %2, %1 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_and_resume_group
func @await_and_resume_group() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateGroup
  %2 = async.runtime.create : !async.group
  // CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
  // CHECK: call @mlirAsyncRuntimeAwaitAllInGroupAndExecute
  // CHECK-SAME: (%[[TOKEN]], %[[HDL]], %[[RESUME]])
  async.runtime.await_and_resume %2, %1 : !async.group
  return
}

// CHECK-LABEL: @resume
func @resume() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
  // CHECK: call @mlirAsyncRuntimeExecute(%[[HDL]], %[[RESUME]])
  async.runtime.resume %1
  return
}

// CHECK-LABEL: @store
func @store() {
  // CHECK: %[[CST:.*]] = constant 1.0
  %0 = constant 1.0 : f32
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %1 = async.runtime.create : !async.value<f32>
  // CHECK: %[[P0:.*]] = call @mlirAsyncRuntimeGetValueStorage(%[[VALUE]])
  // CHECK: %[[P1:.*]] = llvm.bitcast %[[P0]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.store %[[CST]], %[[P1]]
  async.runtime.store %0, %1 : !async.value<f32>
  return
}

// CHECK-LABEL: @load
func @load() -> f32 {
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: %[[P0:.*]] = call @mlirAsyncRuntimeGetValueStorage(%[[VALUE]])
  // CHECK: %[[P1:.*]] = llvm.bitcast %[[P0]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: %[[VALUE:.*]] = llvm.load %[[P1]]
  %1 = async.runtime.load %0 : !async.value<f32>
  // CHECK: return %[[VALUE]] : f32
  return %1 : f32
}

// CHECK-LABEL: @add_token_to_group
func @add_token_to_group() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: %[[GROUP:.*]] = call @mlirAsyncRuntimeCreateGroup
  %1 = async.runtime.create : !async.group
  // CHECK: call @mlirAsyncRuntimeAddTokenToGroup(%[[TOKEN]], %[[GROUP]])
  async.runtime.add_to_group %0, %1 : !async.token
  return
}
