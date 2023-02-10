// RUN: mlir-opt %s -split-input-file -async-to-async-runtime -convert-async-to-llvm='use-opaque-pointers=0' | FileCheck %s



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

// -----

// CHECK-LABEL: execute_no_async_args
func.func @execute_no_async_args(%arg0: f32, %arg1: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn(%arg0, %arg1)
  %token = async.execute {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: call @mlirAsyncRuntimeAwaitToken(%[[TOKEN]])
  // CHECK: %[[IS_ERROR:.*]] = call @mlirAsyncRuntimeIsTokenError(%[[TOKEN]])
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[NOT_ERROR:.*]] = arith.xori %[[IS_ERROR]], %[[TRUE]] : i1
  // CHECK: cf.assert %[[NOT_ERROR]]
  // CHECK-NEXT: return
  async.await %token : !async.token
  return
}

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn(%arg0: f32, %arg1: memref<1xf32>)
// CHECK-SAME: -> !llvm.ptr<i8>

// Create token for return op, and mark a function as a coroutine.
// CHECK: %[[RET:.*]] = call @mlirAsyncRuntimeCreateToken()
// CHECK: %[[HDL:.*]] = llvm.intr.coro.begin

// Pass a suspended coroutine to the async runtime.
// CHECK: %[[STATE:.*]] = llvm.intr.coro.save
// CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
// CHECK: call @mlirAsyncRuntimeExecute(%[[HDL]], %[[RESUME]])
// CHECK: %[[SUSPENDED:.*]] = llvm.intr.coro.suspend %[[STATE]]

// Decide the next block based on the code returned from suspend.
// CHECK: %[[SEXT:.*]] = llvm.sext %[[SUSPENDED]] : i8 to i32
// CHECK: llvm.switch %[[SEXT]] : i32, ^[[SUSPEND:[b0-9]+]]
// CHECK-NEXT: 0: ^[[RESUME:[b0-9]+]]
// CHECK-NEXT: 1: ^[[CLEANUP:[b0-9]+]]

// Resume coroutine after suspension.
// CHECK: ^[[RESUME]]:
// CHECK: memref.store %arg0, %arg1[%c0] : memref<1xf32>
// CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[RET]])

// Delete coroutine.
// CHECK: ^[[CLEANUP]]:
// CHECK: %[[MEM:.*]] = llvm.intr.coro.free
// CHECK: llvm.call @free(%[[MEM]])

// Suspend coroutine, and also a return statement for ramp function.
// CHECK: ^[[SUSPEND]]:
// CHECK: llvm.intr.coro.end
// CHECK: return %[[RET]]

// -----

// CHECK-LABEL: execute_and_return_f32
func.func @execute_and_return_f32() -> f32 {
 // CHECK: %[[RET:.*]]:2 = call @async_execute_fn
  %token, %result = async.execute -> !async.value<f32> {
    %c0 = arith.constant 123.0 : f32
    async.yield %c0 : f32
  }

  // CHECK: %[[STORAGE:.*]] = call @mlirAsyncRuntimeGetValueStorage(%[[RET]]#1)
  // CHECK: %[[ST_F32:.*]] = llvm.bitcast %[[STORAGE]]
  // CHECK: %[[LOADED:.*]] = llvm.load %[[ST_F32]] :  !llvm.ptr<f32>
  %0 = async.await %result : !async.value<f32>

  return %0 : f32
}

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn()
// CHECK: %[[TOKEN:.*]] = call @mlirAsyncRuntimeCreateToken()
// CHECK: %[[VALUE:.*]] = call @mlirAsyncRuntimeCreateValue
// CHECK: %[[HDL:.*]] = llvm.intr.coro.begin

// Suspend coroutine in the beginning.
// CHECK: call @mlirAsyncRuntimeExecute(%[[HDL]],
// CHECK: llvm.intr.coro.suspend

// Emplace result value.
// CHECK: %[[CST:.*]] = arith.constant 1.230000e+02 : f32
// CHECK: %[[STORAGE:.*]] = call @mlirAsyncRuntimeGetValueStorage(%[[VALUE]])
// CHECK: %[[ST_F32:.*]] = llvm.bitcast %[[STORAGE]]
// CHECK: llvm.store %[[CST]], %[[ST_F32]] : !llvm.ptr<f32>
// CHECK: call @mlirAsyncRuntimeEmplaceValue(%[[VALUE]])

// Emplace result token.
// CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[TOKEN]])

// -----

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
