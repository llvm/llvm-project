// RUN: mlir-opt %s -split-input-file -convert-async-to-llvm | FileCheck %s

// CHECK-LABEL: execute_no_async_args
func @execute_no_async_args(%arg0: f32, %arg1: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn(%arg0, %arg1)
  %token = async.execute {
    %c0 = constant 0 : index
    store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: call @mlirAsyncRuntimeAwaitToken(%[[TOKEN]])
  // CHECK-NEXT: return
  async.await %token : !async.token
  return
}

// Function outlined from the async.execute operation.
// CHECK: func @async_execute_fn(%arg0: f32, %arg1: memref<1xf32>)
// CHECK-SAME: -> !llvm.ptr<i8> attributes {sym_visibility = "private"}

// Create token for return op, and mark a function as a coroutine.
// CHECK: %[[RET:.*]] = call @mlirAsyncRuntimeCreateToken()
// CHECK: %[[HDL:.*]] = llvm.call @llvm.coro.begin

// Pass a suspended coroutine to the async runtime.
// CHECK: %[[RESUME:.*]] = llvm.mlir.addressof @__resume
// CHECK: %[[STATE:.*]] = llvm.call @llvm.coro.save
// CHECK: call @mlirAsyncRuntimeExecute(%[[HDL]], %[[RESUME]])
// CHECK: %[[SUSPENDED:.*]] = llvm.call @llvm.coro.suspend(%[[STATE]]

// Decide the next block based on the code returned from suspend.
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8)
// CHECK: %[[NONE:.*]] = llvm.mlir.constant(-1 : i8)
// CHECK: %[[IS_NONE:.*]] = llvm.icmp "eq" %[[SUSPENDED]], %[[NONE]]
// CHECK: llvm.cond_br %[[IS_NONE]], ^[[SUSPEND:.*]], ^[[RESUME_OR_CLEANUP:.*]]

// Decide if branch to resume or cleanup block.
// CHECK: ^[[RESUME_OR_CLEANUP]]:
// CHECK: %[[IS_ZERO:.*]] = llvm.icmp "eq" %[[SUSPENDED]], %[[ZERO]]
// CHECK: llvm.cond_br %[[IS_ZERO]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]

// Resume coroutine after suspension.
// CHECK: ^[[RESUME]]:
// CHECK: store %arg0, %arg1[%c0] : memref<1xf32>
// CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[RET]])

// Delete coroutine.
// CHECK: ^[[CLEANUP]]:
// CHECK: %[[MEM:.*]] = llvm.call @llvm.coro.free
// CHECK: llvm.call @free(%[[MEM]])

// Suspend coroutine, and also a return statement for ramp function.
// CHECK: ^[[SUSPEND]]:
// CHECK: llvm.call @llvm.coro.end
// CHECK: return %[[RET]]

// -----

// CHECK-LABEL: nested_async_execute
func @nested_async_execute(%arg0: f32, %arg1: f32, %arg2: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn_0(%arg0, %arg2, %arg1)
  %token0 = async.execute {
    %c0 = constant 0 : index

    %token1 = async.execute {
      %c1 = constant 1: index
      store %arg0, %arg2[%c0] : memref<1xf32>
      async.yield
    }
    async.await %token1 : !async.token

    store %arg1, %arg2[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: call @mlirAsyncRuntimeAwaitToken(%[[TOKEN]])
  // CHECK-NEXT: return
  async.await %token0 : !async.token
  return
}

// Function outlined from the inner async.execute operation.
// CHECK: func @async_execute_fn(%arg0: f32, %arg1: memref<1xf32>, %arg2: index)
// CHECK-SAME: -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
// CHECK: %[[RET_0:.*]] = call @mlirAsyncRuntimeCreateToken()
// CHECK: %[[HDL_0:.*]] = llvm.call @llvm.coro.begin
// CHECK: call @mlirAsyncRuntimeExecute
// CHECK: llvm.call @llvm.coro.suspend
// CHECK: store %arg0, %arg1[%arg2] : memref<1xf32>
// CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[RET_0]])

// Function outlined from the outer async.execute operation.
// CHECK: func @async_execute_fn_0(%arg0: f32, %arg1: memref<1xf32>, %arg2: f32)
// CHECK-SAME: -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
// CHECK: %[[RET_1:.*]] = call @mlirAsyncRuntimeCreateToken()
// CHECK: %[[HDL_1:.*]] = llvm.call @llvm.coro.begin

// Suspend coroutine in the beginning.
// CHECK: call @mlirAsyncRuntimeExecute
// CHECK: llvm.call @llvm.coro.suspend

// Suspend coroutine second time waiting for the completion of inner execute op.
// CHECK: %[[TOKEN_1:.*]] = call @async_execute_fn
// CHECK: llvm.call @llvm.coro.save
// CHECK: call @mlirAsyncRuntimeAwaitTokenAndExecute(%[[TOKEN_1]], %[[HDL_1]]
// CHECK: llvm.call @llvm.coro.suspend

// Emplace result token after second resumption.
// CHECK: store %arg2, %arg1[%c0] : memref<1xf32>
// CHECK: call @mlirAsyncRuntimeEmplaceToken(%[[RET_1]])


