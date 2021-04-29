// RUN: mlir-opt %s -async-runtime-ref-counting | FileCheck %s

// CHECK-LABEL: @token
func private @token() -> !async.token

// CHECK-LABEL: @cond
func private @cond() -> i1

// CHECK-LABEL: @take_token
func private @take_token(%arg0: !async.token)

// CHECK-LABEL: @token_arg_no_uses
// CHECK: %[[TOKEN:.*]]: !async.token
func @token_arg_no_uses(%arg0: !async.token) {
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  return
}

// CHECK-LABEL: @token_value_no_uses
func @token_value_no_uses() {
  // CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  %0 = async.runtime.create : !async.token
  return
}

// CHECK-LABEL: @token_returned_no_uses
func @token_returned_no_uses() {
  // CHECK: %[[TOKEN:.*]] = call @token
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  %0 = call @token() : () -> !async.token
  return
}

// CHECK-LABEL: @token_arg_to_func
// CHECK: %[[TOKEN:.*]]: !async.token
func @token_arg_to_func(%arg0: !async.token) {
  // CHECK: async.runtime.add_ref %[[TOKEN]] {count = 1 : i32} : !async.token
  call @take_token(%arg0): (!async.token) -> ()
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32} : !async.token
  return
}

// CHECK-LABEL: @token_value_to_func
func @token_value_to_func() {
  // CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
  %0 = async.runtime.create : !async.token
  // CHECK: async.runtime.add_ref %[[TOKEN]] {count = 1 : i32} : !async.token
  call @take_token(%0): (!async.token) -> ()
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  return
}

// CHECK-LABEL: @token_arg_cond_br_await_with_fallthough
// CHECK: %[[TOKEN:.*]]: !async.token
func @token_arg_cond_br_await_with_fallthough(%arg0: !async.token, %arg1: i1) {
  // CHECK: cond_br
  // CHECK-SAME: ^[[BB1:.*]], ^[[BB2:.*]]
  cond_br %arg1, ^bb1, ^bb2
^bb1:
  // CHECK: ^[[BB1]]:
  // CHECK:   br ^[[BB2]]
  br ^bb2
^bb2:
  // CHECK: ^[[BB2]]:
  // CHECK:   async.runtime.await %[[TOKEN]]
  // CHECK:   async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.runtime.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @token_simple_return
func @token_simple_return() -> !async.token {
  // CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
  %token = async.runtime.create : !async.token
  // CHECK: return %[[TOKEN]]
  return %token : !async.token
}

// CHECK-LABEL: @token_coro_return
// CHECK-NOT: async.runtime.drop_ref
// CHECK-NOT: async.runtime.add_ref
func @token_coro_return() -> !async.token {
  %token = async.runtime.create : !async.token
  %id = async.coro.id
  %hdl = async.coro.begin %id
  %saved = async.coro.save %hdl
  async.runtime.resume %hdl
  async.coro.suspend %saved, ^suspend, ^resume, ^cleanup
^resume:
  br ^cleanup
^cleanup:
  async.coro.free %id, %hdl
  br ^suspend
^suspend:
  async.coro.end %hdl
  return %token : !async.token
}

// CHECK-LABEL: @token_coro_await_and_resume
// CHECK: %[[TOKEN:.*]]: !async.token
func @token_coro_await_and_resume(%arg0: !async.token) -> !async.token {
  %token = async.runtime.create : !async.token
  %id = async.coro.id
  %hdl = async.coro.begin %id
  %saved = async.coro.save %hdl
  // CHECK: async.runtime.await_and_resume %[[TOKEN]]
  async.runtime.await_and_resume %arg0, %hdl : !async.token
  // CHECK-NEXT: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.coro.suspend %saved, ^suspend, ^resume, ^cleanup
^resume:
  br ^cleanup
^cleanup:
  async.coro.free %id, %hdl
  br ^suspend
^suspend:
  async.coro.end %hdl
  return %token : !async.token
}

// CHECK-LABEL: @value_coro_await_and_resume
// CHECK: %[[VALUE:.*]]: !async.value<f32>
func @value_coro_await_and_resume(%arg0: !async.value<f32>) -> !async.token {
  %token = async.runtime.create : !async.token
  %id = async.coro.id
  %hdl = async.coro.begin %id
  %saved = async.coro.save %hdl
  // CHECK: async.runtime.await_and_resume %[[VALUE]]
  async.runtime.await_and_resume %arg0, %hdl : !async.value<f32>
  // CHECK: async.coro.suspend
  // CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]
  async.coro.suspend %saved, ^suspend, ^resume, ^cleanup
^resume:
  // CHECK: ^[[RESUME]]:
  // CHECK:   %[[LOADED:.*]] = async.runtime.load %[[VALUE]]
  // CHECK:   async.runtime.drop_ref %[[VALUE]] {count = 1 : i32}
  %0 = async.runtime.load %arg0 : !async.value<f32>
  // CHECK:  addf %[[LOADED]], %[[LOADED]]
  %1 = addf %0, %0 : f32
  br ^cleanup
^cleanup:
  async.coro.free %id, %hdl
  br ^suspend
^suspend:
  async.coro.end %hdl
  return %token : !async.token
}

// CHECK-LABEL: @outlined_async_execute
// CHECK: %[[TOKEN:.*]]: !async.token
func private @outlined_async_execute(%arg0: !async.token) -> !async.token {
  %0 = async.runtime.create : !async.token
  %1 = async.coro.id
  %2 = async.coro.begin %1
  %3 = async.coro.save %2
  async.runtime.resume %2
  // CHECK: async.coro.suspend
  async.coro.suspend %3, ^suspend, ^resume, ^cleanup
^resume:
  // CHECK: ^[[RESUME:.*]]:
  %4 = async.coro.save %2
  async.runtime.await_and_resume %arg0, %2 : !async.token
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: async.coro.suspend
  async.coro.suspend %4, ^suspend, ^resume_1, ^cleanup
^resume_1:
  // CHECK: ^[[RESUME_1:.*]]:
  // CHECK:   async.runtime.set_available
  async.runtime.set_available %0 : !async.token
  br ^cleanup
^cleanup:
  // CHECK: ^[[CLEANUP:.*]]:
  // CHECK:   async.coro.free
  async.coro.free %1, %2
  br ^suspend
^suspend:
  // CHECK: ^[[SUSPEND:.*]]:
  // CHECK:   async.coro.end
  async.coro.end %2
  return %0 : !async.token
}

// CHECK-LABEL: @token_await_inside_nested_region
// CHECK: %[[ARG:.*]]: i1
func @token_await_inside_nested_region(%arg0: i1) {
  // CHECK: %[[TOKEN:.*]] = call @token()
  %token = call @token() : () -> !async.token
  // CHECK: scf.if %[[ARG]] {
  scf.if %arg0 {
    // CHECK: async.runtime.await %[[TOKEN]]
    async.runtime.await %token : !async.token
  }
  // CHECK: }
  // CHECK: async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: return
  return
}

// CHECK-LABEL: @token_defined_in_the_loop
func @token_defined_in_the_loop() {
  br ^bb1
^bb1:
  // CHECK: ^[[BB1:.*]]:
  // CHECK:   %[[TOKEN:.*]] = call @token()
  %token = call @token() : () -> !async.token
  // CHECK:   async.runtime.await %[[TOKEN]]
  // CHECK:   async.runtime.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.runtime.await %token : !async.token
  %0 = call @cond(): () -> (i1)
  cond_br %0, ^bb1, ^bb2
^bb2:
  // CHECK: ^[[BB2:.*]]:
  // CHECK:   return
  return
}
