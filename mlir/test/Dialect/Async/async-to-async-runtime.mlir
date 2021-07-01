// RUN: mlir-opt %s -split-input-file -async-to-async-runtime                  \
// RUN:   | FileCheck %s --dump-input=always

// CHECK-LABEL: @execute_no_async_args
func @execute_no_async_args(%arg0: f32, %arg1: memref<1xf32>) {
  %token = async.execute {
    %c0 = constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  async.await %token : !async.token
  return
}

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn
// CHECK-SAME: -> !async.token

// Create token for return op, and mark a function as a coroutine.
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin

// Pass a suspended coroutine to the async runtime.
// CHECK: %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]

// Resume coroutine after suspension.
// CHECK: ^[[RESUME]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]

// Delete coroutine.
// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]

// Suspend coroutine, and also a return statement for ramp function.
// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]]

// -----

// CHECK-LABEL: @nested_async_execute
func @nested_async_execute(%arg0: f32, %arg1: f32, %arg2: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn_0(%arg0, %arg2, %arg1)
  %token0 = async.execute {
    %c0 = constant 0 : index

    %token1 = async.execute {
      %c1 = constant 1: index
      memref.store %arg0, %arg2[%c0] : memref<1xf32>
      async.yield
    }
    async.await %token1 : !async.token

    memref.store %arg1, %arg2[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: async.runtime.await %[[TOKEN]]
  // CHECK-NEXT: return
  async.await %token0 : !async.token
  return
}

// Function outlined from the inner async.execute operation.
// CHECK-LABEL: func private @async_execute_fn
// CHECK-SAME: -> !async.token

// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin

// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]

// CHECK: ^[[RESUME]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]

// Function outlined from the outer async.execute operation.
// CHECK-LABEL: func private @async_execute_fn_0
// CHECK-SAME: -> !async.token

// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin

// Suspend coroutine in the beginning.
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_0:.*]], ^[[CLEANUP:.*]]

// Suspend coroutine second time waiting for the completion of inner execute op.
// CHECK: ^[[RESUME_0]]:
// CHECK:   %[[INNER_TOKEN:.*]] = call @async_execute_fn
// CHECK:   %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK:   async.runtime.await_and_resume %[[INNER_TOKEN]], %[[HDL]]
// CHECK:   async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND]], ^[[RESUME_1:.*]], ^[[CLEANUP]]

// Check the error of the awaited token after resumption.
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[ERR:.*]] = async.runtime.is_error %[[INNER_TOKEN]]
// CHECK:   cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// Set token available if the token is not in the error state.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[SET_ERROR]]:
// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @async_execute_token_dependency
func @async_execute_token_dependency(%arg0: f32, %arg1: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn
  %token = async.execute {
    %c0 = constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: call @async_execute_fn_0(%[[TOKEN]], %arg0, %arg1)
  %token_0 = async.execute [%token] {
    %c0 = constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  return
}

// Function outlined from the first async.execute operation.
// CHECK-LABEL: func private @async_execute_fn
// CHECK-SAME: -> !async.token
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: return %[[TOKEN]] : !async.token

// Function outlined from the second async.execute operation with dependency.
// CHECK-LABEL: func private @async_execute_fn_0
// CHECK-SAME:    %[[ARG0:.*]]: !async.token
// CHECK-SAME:    %[[ARG1:.*]]: f32
// CHECK-SAME:    %[[ARG2:.*]]: memref<1xf32>
// CHECK-SAME: -> !async.token
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[HDL:.*]] = async.coro.begin

// Suspend coroutine in the beginning.
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_0:.*]], ^[[CLEANUP:.*]]

// Suspend coroutine second time waiting for the completion of token dependency.
// CHECK: ^[[RESUME_0]]:
// CHECK:   %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK:   async.runtime.await_and_resume %[[ARG0]], %[[HDL]]
// CHECK:   async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND]], ^[[RESUME_1:.*]], ^[[CLEANUP]]

// Check the error of the awaited token after resumption.
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[ERR:.*]] = async.runtime.is_error %[[ARG0]]
// CHECK:   cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// Emplace result token after second resumption and error checking.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @async_group_await_all
func @async_group_await_all(%arg0: f32, %arg1: memref<1xf32>) {
  // CHECK: %[[C:.*]] = constant 1 : index
  %c = constant 1 : index
  // CHECK: %[[GROUP:.*]] = async.runtime.create_group %[[C]] : !async.group
  %0 = async.create_group %c : !async.group

  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn
  %token = async.execute { async.yield }
  // CHECK: async.runtime.add_to_group %[[TOKEN]], %[[GROUP]]
  async.add_to_group %token, %0 : !async.token

  // CHECK: call @async_execute_fn_0
  async.execute {
    async.await_all %0
    async.yield
  }

  // CHECK: async.runtime.await %[[GROUP]] : !async.group
  async.await_all %0
  return
}

// Function outlined from the second async.execute operation.
// CHECK-LABEL: func private @async_execute_fn_0
// CHECK-SAME: (%[[ARG:.*]]: !async.group) -> !async.token

// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[HDL:.*]] = async.coro.begin

// Suspend coroutine in the beginning.
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_0:.*]], ^[[CLEANUP:.*]]

// Suspend coroutine second time waiting for the group.
// CHECK: ^[[RESUME_0]]:
// CHECK:   async.runtime.await_and_resume %[[ARG]], %[[HDL]]
// CHECK:   async.coro.suspend
// CHECK-SAME: ^[[SUSPEND]], ^[[RESUME_1:.*]], ^[[CLEANUP]]

// Check the error of the awaited token after resumption.
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[ERR:.*]] = async.runtime.is_error %[[ARG]]
// CHECK:   cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// Emplace result token after error checking.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @execute_and_return_f32
func @execute_and_return_f32() -> f32 {
 // CHECK: %[[RET:.*]]:2 = call @async_execute_fn
  %token, %result = async.execute -> !async.value<f32> {
    %c0 = constant 123.0 : f32
    async.yield %c0 : f32
  }

  // CHECK: async.runtime.await %[[RET]]#1 : !async.value<f32>
  // CHECK: %[[VALUE:.*]] = async.runtime.load %[[RET]]#1 : !async.value<f32>
  %0 = async.await %result : !async.value<f32>

  // CHECK: return %[[VALUE]]
  return %0 : f32
}

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn()
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[VALUE:.*]] = async.runtime.create : !async.value<f32>
// CHECK: %[[HDL:.*]] = async.coro.begin

// Suspend coroutine in the beginning.
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]

// Emplace result value.
// CHECK: ^[[RESUME]]:
// CHECK:   %[[CST:.*]] = constant 1.230000e+02 : f32
// CHECK:   async.runtime.store %cst, %[[VALUE]]
// CHECK:   async.runtime.set_available %[[VALUE]]
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @async_value_operands
func @async_value_operands() {
  // CHECK: %[[RET:.*]]:2 = call @async_execute_fn
  %token, %result = async.execute -> !async.value<f32> {
    %c0 = constant 123.0 : f32
    async.yield %c0 : f32
  }

  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn_0(%[[RET]]#1)
  %token0 = async.execute(%result as %value: !async.value<f32>) {
    %0 = addf %value, %value : f32
    async.yield
  }

  // CHECK: async.runtime.await %[[TOKEN]] : !async.token
  async.await %token0 : !async.token

  return
}

// Function outlined from the first async.execute operation.
// CHECK-LABEL: func private @async_execute_fn()

// Function outlined from the second async.execute operation.
// CHECK-LABEL: func private @async_execute_fn_0
// CHECK-SAME: (%[[ARG:.*]]: !async.value<f32>) -> !async.token

// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[HDL:.*]] = async.coro.begin

// Suspend coroutine in the beginning.
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_0:.*]], ^[[CLEANUP:.*]]

// Suspend coroutine second time waiting for the async operand.
// CHECK: ^[[RESUME_0]]:
// CHECK:   async.runtime.await_and_resume %[[ARG]], %[[HDL]]
// CHECK:   async.coro.suspend
// CHECK-SAME: ^[[SUSPEND]], ^[[RESUME_1:.*]], ^[[CLEANUP]]

// Check the error of the awaited token after resumption.
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[ERR:.*]] = async.runtime.is_error %[[ARG]]
// CHECK:   cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// // Load from the async.value argument after error checking.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   %[[LOADED:.*]] = async.runtime.load %[[ARG]] : !async.value<f32
// CHECK:   addf %[[LOADED]], %[[LOADED]] : f32
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @execute_asserttion
func @execute_asserttion(%arg0: i1) {
  %token = async.execute {
    assert %arg0, "error"
    async.yield
  }
  async.await %token : !async.token
  return
}

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn(
// CHECK-SAME:  %[[ARG0:.*]]: i1
// CHECK-SAME:  -> !async.token

// Create token for return op, and mark a function as a coroutine.
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin

// Initial coroutine suspension.
// CHECK:      async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]

// Resume coroutine after suspension.
// CHECK: ^[[RESUME]]:
// CHECK:   cond_br %[[ARG0]], ^[[SET_AVAILABLE:.*]], ^[[SET_ERROR:.*]]

// Set coroutine completion token to available state.
// CHECK: ^[[SET_AVAILABLE]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]

// Set coroutine completion token to error state.
// CHECK: ^[[SET_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]

// Delete coroutine.
// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]

// Suspend coroutine, and also a return statement for ramp function.
// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]]
