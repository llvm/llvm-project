// RUN: mlir-opt %s -split-input-file -async-func-to-async-runtime             \
// RUN:   -async-to-async-runtime | FileCheck %s --dump-input=always

// CHECK-LABEL: @execute_no_async_args
func.func @execute_no_async_args(%arg0: f32, %arg1: memref<1xf32>) {
  %token = async.execute {
    %c0 = arith.constant 0 : index
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
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[DESTROY:.*]]

// Resume coroutine after suspension.
// CHECK: ^[[RESUME]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   cf.br ^[[CLEANUP:.*]]

// Delete coroutine.
// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]

// Delete coroutine.
// CHECK: ^[[DESTROY]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]

// Suspend coroutine, and also a return statement for ramp function.
// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]]

// -----

// CHECK-LABEL: @nested_async_execute
func.func @nested_async_execute(%arg0: f32, %arg1: f32, %arg2: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn_0(%arg0, %arg2, %arg1)
  %token0 = async.execute {
    %c0 = arith.constant 0 : index

    %token1 = async.execute {
      %c1 = arith.constant 1: index
      memref.store %arg0, %arg2[%c0] : memref<1xf32>
      async.yield
    }
    async.await %token1 : !async.token

    memref.store %arg1, %arg2[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: async.runtime.await %[[TOKEN]]
  // CHECK: %[[IS_ERROR:.*]] = async.runtime.is_error %[[TOKEN]]
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[NOT_ERROR:.*]] = arith.xori %[[IS_ERROR]], %[[TRUE]] : i1
  // CHECK: cf.assert %[[NOT_ERROR]]
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
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[DESTROY:.*]]

// CHECK: ^[[RESUME]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   cf.br ^[[CLEANUP:.*]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[DESTROY]]:

// Function outlined from the outer async.execute operation.
// CHECK-LABEL: func private @async_execute_fn_0
// CHECK-SAME: -> !async.token

// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin

// Suspend coroutine in the beginning.
// CHECK: async.runtime.resume %[[HDL]]
// CHECK: async.coro.suspend
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_0:.*]], ^[[DESTROY_0:.*]]

// Suspend coroutine second time waiting for the completion of inner execute op.
// CHECK: ^[[RESUME_0]]:
// CHECK:   %[[INNER_TOKEN:.*]] = call @async_execute_fn
// CHECK:   %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK:   async.runtime.await_and_resume %[[INNER_TOKEN]], %[[HDL]]
// CHECK:   async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND]], ^[[RESUME_1:.*]], ^[[DESTROY_0]]

// Check the error of the awaited token after resumption.
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[ERR:.*]] = async.runtime.is_error %[[INNER_TOKEN]]
// CHECK:   cf.cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// Set token available if the token is not in the error state.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   cf.br ^[[CLEANUP_0:.*]]

// CHECK: ^[[SET_ERROR]]:
// CHECK: ^[[CLEANUP_0]]:
// CHECK: ^[[DESTROY_0]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @async_execute_token_dependency
func.func @async_execute_token_dependency(%arg0: f32, %arg1: memref<1xf32>) {
  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn
  %token = async.execute {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  // CHECK: call @async_execute_fn_0(%[[TOKEN]], %arg0, %arg1)
  %token_0 = async.execute [%token] {
    %c0 = arith.constant 0 : index
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
// CHECK:   cf.cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// Emplace result token after second resumption and error checking.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   memref.store
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @async_group_await_all
func.func @async_group_await_all(%arg0: f32, %arg1: memref<1xf32>) {
  // CHECK: %[[C:.*]] = arith.constant 1 : index
  %c = arith.constant 1 : index
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
// CHECK:   cf.cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// Emplace result token after error checking.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @execute_and_return_f32
func.func @execute_and_return_f32() -> f32 {
 // CHECK: %[[RET:.*]]:2 = call @async_execute_fn
  %token, %result = async.execute -> !async.value<f32> {
    %c0 = arith.constant 123.0 : f32
    async.yield %c0 : f32
  }

  // CHECK: async.runtime.await %[[RET]]#1 : !async.value<f32>
  // CHECK: %[[VALUE:.*]] = async.runtime.load %[[RET]]#1 : <f32>
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
// CHECK:   %[[CST:.*]] = arith.constant 1.230000e+02 : f32
// CHECK:   async.runtime.store %cst, %[[VALUE]]
// CHECK:   async.runtime.set_available %[[VALUE]]
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @async_value_operands
func.func @async_value_operands() {
  // CHECK: %[[RET:.*]]:2 = call @async_execute_fn
  %token, %result = async.execute -> !async.value<f32> {
    %c0 = arith.constant 123.0 : f32
    async.yield %c0 : f32
  }

  // CHECK: %[[TOKEN:.*]] = call @async_execute_fn_0(%[[RET]]#1)
  %token0 = async.execute(%result as %value: !async.value<f32>) {
    %0 = arith.addf %value, %value : f32
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
// CHECK:   cf.cond_br %[[ERR]], ^[[SET_ERROR:.*]], ^[[CONTINUATION:.*]]

// // Load from the async.value argument after error checking.
// CHECK: ^[[CONTINUATION:.*]]:
// CHECK:   %[[LOADED:.*]] = async.runtime.load %[[ARG]] : <f32
// CHECK:   arith.addf %[[LOADED]], %[[LOADED]] : f32
// CHECK:   async.runtime.set_available %[[TOKEN]]

// CHECK: ^[[CLEANUP]]:
// CHECK: ^[[SUSPEND]]:

// -----

// CHECK-LABEL: @execute_assertion
func.func @execute_assertion(%arg0: i1) {
  %token = async.execute {
    cf.assert %arg0, "error"
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
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[DESTROY:.*]]

// Resume coroutine after suspension.
// CHECK: ^[[RESUME]]:
// CHECK:   cf.cond_br %[[ARG0]], ^[[SET_AVAILABLE:.*]], ^[[SET_ERROR:.*]]

// Set coroutine completion token to available state.
// CHECK: ^[[SET_AVAILABLE]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   cf.br ^[[CLEANUP:.*]]

// Set coroutine completion token to error state.
// CHECK: ^[[SET_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   cf.br ^[[CLEANUP]]

// Delete coroutine.
// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]

// Delete coroutine.
// CHECK: ^[[DESTROY]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]

// Suspend coroutine, and also a return statement for ramp function.
// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]]

// -----
// Structured control flow operations with async operations in the body must be
// lowered to branch-based control flow to enable coroutine CFG rewrite.

// CHECK-LABEL: @lower_scf_to_cfg
func.func @lower_scf_to_cfg(%arg0: f32, %arg1: memref<1xf32>, %arg2: i1) {
  %token0 = async.execute { async.yield }
  %token1 = async.execute {
    scf.if %arg2 {
      async.await %token0 : !async.token
    } else {
      async.await %token0 : !async.token
    }
    async.yield
  }
  return
}

// Function outlined from the first async.execute operation.
// CHECK-LABEL: func private @async_execute_fn(
// CHECK-SAME: -> !async.token

// Function outlined from the second async.execute operation.
// CHECK-LABEL: func private @async_execute_fn_0(
// CHECK:         %[[TOKEN:.*]]: !async.token
// CHECK:         %[[FLAG:.*]]: i1
// CHECK-SAME: -> !async.token

// Check that structured control flow lowered to CFG.
// CHECK-NOT: scf.if
// CHECK: cf.cond_br %[[FLAG]]

// -----
// Constants captured by the async.execute region should be cloned into the
// outline async execute function.

// CHECK-LABEL: @clone_constants
func.func @clone_constants(%arg0: f32, %arg1: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %token = async.execute {
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  async.await %token : !async.token
  return
}

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn(
// CHECK-SAME:    %[[VALUE:arg[0-9]+]]: f32,
// CHECK-SAME:    %[[MEMREF:arg[0-9]+]]: memref<1xf32>
// CHECK-SAME:  ) -> !async.token
// CHECK:         %[[CST:.*]] = arith.constant 0 : index
// CHECK:         memref.store %[[VALUE]], %[[MEMREF]][%[[CST]]]

// -----
// Async Functions should be none blocking

// CHECK-LABEL: @async_func_await
async.func @async_func_await(%arg0: f32, %arg1: !async.value<f32>)
              -> !async.token {
  %0 = async.await %arg1 : !async.value<f32>
  return
}
// Create token for return op, and mark a function as a coroutine.
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin
// CHECK:   cf.br ^[[ORIGIN_ENTRY:.*]]

// CHECK: ^[[ORIGIN_ENTRY]]:
// CHECK: %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[arg1:.*]], %[[HDL]] :
// CHECK-SAME: !async.value<f32>
// CHECK: async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]

// -----
// Async execute inside async func

// CHECK-LABEL: @execute_in_async_func
async.func @execute_in_async_func(%arg0: f32, %arg1: memref<1xf32>)
             -> !async.token {
  %token = async.execute {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32>
    async.yield
  }
  async.await %token : !async.token
  return
}
// Call outlind async execute Function
// CHECK: %[[RES:.*]] = call @async_execute_fn(
// CHECK-SAME:    %[[VALUE:arg[0-9]+]],
// CHECK-SAME:    %[[MEMREF:arg[0-9]+]]
// CHECK-SAME:  ) : (f32, memref<1xf32>) -> !async.token

// Function outlined from the async.execute operation.
// CHECK-LABEL: func private @async_execute_fn(
// CHECK-SAME:    %[[VALUE:arg[0-9]+]]: f32,
// CHECK-SAME:    %[[MEMREF:arg[0-9]+]]: memref<1xf32>
// CHECK-SAME:  ) -> !async.token
// CHECK:         %[[CST:.*]] = arith.constant 0 : index
// CHECK:         memref.store %[[VALUE]], %[[MEMREF]][%[[CST]]]
