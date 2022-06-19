// RUN: mlir-opt %s -convert-async-to-llvm | FileCheck %s --dump-input=always

// CHECK-LABEL: @create_token
func.func @create_token() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  return
}

// CHECK-LABEL: @create_value
func.func @create_value() {
  // CHECK: %[[NULL:.*]] = llvm.mlir.null : !llvm.ptr<f32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[OFFSET:.*]] = llvm.getelementptr %[[NULL]][%[[ONE]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[OFFSET]]
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue(%[[SIZE]])
  %0 = async.runtime.create : !async.value<f32>
  return
}

// CHECK-LABEL: @create_group
func.func @create_group() {
  // CHECK: %[[C:.*]] = arith.constant 1 : index
  // CHECK: %[[S:.*]] = builtin.unrealized_conversion_cast %[[C]] : index to i64
  %c = arith.constant 1 : index
  // CHECK: %[[GROUP:.*]] = call @mlirAsyncRuntimeCreateGroup(%[[S]])
  %0 = async.runtime.create_group  %c: !async.group
  return
}

// CHECK-LABEL: @set_token_available
func.func @set_token_available() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[TOKEN]])
  async.runtime.set_available %0 : !async.token
  return
}

// CHECK-LABEL: @set_value_available
func.func @set_value_available() {
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: call @mlirAsyncRuntimeEmplaceValue(%[[VALUE]])
  async.runtime.set_available %0 : !async.value<f32>
  return
}

// CHECK-LABEL: @is_token_error
func.func @is_token_error() -> i1 {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: %[[ERR:.*]] = call @mlirAsyncRuntimeIsTokenError(%[[TOKEN]])
  %1 = async.runtime.is_error %0 : !async.token
  return %1 : i1
}

// CHECK-LABEL: @is_value_error
func.func @is_value_error() -> i1 {
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: %[[ERR:.*]] = call @mlirAsyncRuntimeIsValueError(%[[VALUE]])
  %1 = async.runtime.is_error %0 : !async.value<f32>
  return %1 : i1
}

// CHECK-LABEL: @await_token
func.func @await_token() {
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: call @mlirAsyncRuntimeAwaitToken(%[[TOKEN]])
  async.runtime.await %0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func.func @await_value() {
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %0 = async.runtime.create : !async.value<f32>
  // CHECK: call @mlirAsyncRuntimeAwaitValue(%[[VALUE]])
  async.runtime.await %0 : !async.value<f32>
  return
}

// CHECK-LABEL: @await_group
func.func @await_group() {
  %c = arith.constant 1 : index
  // CHECK: %[[GROUP:.*]] = call @mlirAsyncRuntimeCreateGroup
  %0 = async.runtime.create_group %c: !async.group
  // CHECK: call @mlirAsyncRuntimeAwaitAllInGroup(%[[GROUP]])
  async.runtime.await %0 : !async.group
  return
}

// CHECK-LABEL: @await_and_resume_token
func.func @await_and_resume_token() {
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
func.func @await_and_resume_value() {
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
func.func @await_and_resume_group() {
  %c = arith.constant 1 : index
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateGroup
  %2 = async.runtime.create_group %c : !async.group
  // CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
  // CHECK: call @mlirAsyncRuntimeAwaitAllInGroupAndExecute
  // CHECK-SAME: (%[[TOKEN]], %[[HDL]], %[[RESUME]])
  async.runtime.await_and_resume %2, %1 : !async.group
  return
}

// CHECK-LABEL: @resume
func.func @resume() {
  %0 = async.coro.id
  // CHECK: %[[HDL:.*]] = llvm.intr.coro.begin
  %1 = async.coro.begin %0
  // CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
  // CHECK: call @mlirAsyncRuntimeExecute(%[[HDL]], %[[RESUME]])
  async.runtime.resume %1
  return
}

// CHECK-LABEL: @store
func.func @store() {
  // CHECK: %[[CST:.*]] = arith.constant 1.0
  %0 = arith.constant 1.0 : f32
  // CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
  %1 = async.runtime.create : !async.value<f32>
  // CHECK: %[[P0:.*]] = call @mlirAsyncRuntimeGetValueStorage(%[[VALUE]])
  // CHECK: %[[P1:.*]] = llvm.bitcast %[[P0]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.store %[[CST]], %[[P1]]
  async.runtime.store %0, %1 : !async.value<f32>
  return
}

// CHECK-LABEL: @load
func.func @load() -> f32 {
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
func.func @add_token_to_group() {
  %c = arith.constant 1 : index
  // CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken
  %0 = async.runtime.create : !async.token
  // CHECK: %[[GROUP:.*]] = call @mlirAsyncRuntimeCreateGroup
  %1 = async.runtime.create_group %c : !async.group
  // CHECK: call @mlirAsyncRuntimeAddTokenToGroup(%[[TOKEN]], %[[GROUP]])
  async.runtime.add_to_group %0, %1 : !async.token
  return
}
