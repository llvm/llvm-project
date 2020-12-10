// RUN: mlir-opt %s -async-ref-counting | FileCheck %s

// CHECK-LABEL: @cond
func private @cond() -> i1

// CHECK-LABEL: @token_arg_no_uses
func @token_arg_no_uses(%arg0: !async.token) {
  // CHECK: async.drop_ref %arg0 {count = 1 : i32}
  return
}

// CHECK-LABEL: @token_arg_conditional_await
func @token_arg_conditional_await(%arg0: !async.token, %arg1: i1) {
  cond_br %arg1, ^bb1, ^bb2
^bb1:
  // CHECK: async.drop_ref %arg0 {count = 1 : i32}
  return
^bb2:
  // CHECK: async.await %arg0
  // CHECK: async.drop_ref %arg0 {count = 1 : i32}
  async.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @token_no_uses
func @token_no_uses() {
  // CHECK: %[[TOKEN:.*]] = async.execute
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  %token = async.execute {
    async.yield
  }
  return
}

// CHECK-LABEL: @token_return
func @token_return() -> !async.token {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  // CHECK: return %[[TOKEN]]
  return %token : !async.token
}

// CHECK-LABEL: @token_await
func @token_await() {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  // CHECK: async.await %[[TOKEN]]
  async.await %token : !async.token
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: return
  return
}

// CHECK-LABEL: @token_await_and_return
func @token_await_and_return() -> !async.token {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  // CHECK: async.await %[[TOKEN]]
  // CHECK-NOT: async.drop_ref
  async.await %token : !async.token
  // CHECK: return %[[TOKEN]]
  return %token : !async.token
}

// CHECK-LABEL: @token_await_inside_scf_if
func @token_await_inside_scf_if(%arg0: i1) {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  // CHECK: scf.if %arg0 {
  scf.if %arg0 {
    // CHECK: async.await %[[TOKEN]]
    async.await %token : !async.token
  }
  // CHECK: }
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: return
  return
}

// CHECK-LABEL: @token_conditional_await
func @token_conditional_await(%arg0: i1) {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  return
^bb2:
  // CHECK: async.await %[[TOKEN]]
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.await %token : !async.token
  return
}

// CHECK-LABEL: @token_await_in_the_loop
func @token_await_in_the_loop() {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  br ^bb1
^bb1:
  // CHECK: async.await %[[TOKEN]]
  async.await %token : !async.token
  %0 = call @cond(): () -> (i1)
  cond_br %0, ^bb1, ^bb2
^bb2:
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  return
}

// CHECK-LABEL: @token_defined_in_the_loop
func @token_defined_in_the_loop() {
  br ^bb1
^bb1:
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }
  // CHECK: async.await %[[TOKEN]]
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.await %token : !async.token
  %0 = call @cond(): () -> (i1)
  cond_br %0, ^bb1, ^bb2
^bb2:
  return
}

// CHECK-LABEL: @token_capture
func @token_capture() {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }

  // CHECK: async.add_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: %[[TOKEN_0:.*]] = async.execute
  %token_0 = async.execute {
    // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
    // CHECK-NEXT: async.yield
    async.await %token : !async.token
    async.yield
  }
  // CHECK: async.drop_ref %[[TOKEN_0]] {count = 1 : i32}
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: return
  return
}

// CHECK-LABEL: @token_nested_capture
func @token_nested_capture() {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }

  // CHECK: async.add_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: %[[TOKEN_0:.*]] = async.execute
  %token_0 = async.execute {
    // CHECK: async.add_ref %[[TOKEN]] {count = 1 : i32}
    // CHECK: %[[TOKEN_1:.*]] = async.execute
    %token_1 = async.execute {
      // CHECK: async.add_ref %[[TOKEN]] {count = 1 : i32}
      // CHECK: %[[TOKEN_2:.*]] = async.execute
      %token_2 = async.execute {
        // CHECK: async.await %[[TOKEN]]
        // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
        async.await %token : !async.token
        async.yield
      }
      // CHECK: async.drop_ref %[[TOKEN_2]] {count = 1 : i32}
      // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
      async.yield
    }
    // CHECK: async.drop_ref %[[TOKEN_1]] {count = 1 : i32}
    // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
    async.yield
  }
  // CHECK: async.drop_ref %[[TOKEN_0]] {count = 1 : i32}
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: return
  return
}

// CHECK-LABEL: @token_dependency
func @token_dependency() {
  // CHECK: %[[TOKEN:.*]] = async.execute
  %token = async.execute {
    async.yield
  }

  // CHECK: async.add_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: %[[TOKEN_0:.*]] = async.execute
  %token_0 = async.execute[%token] {
    // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
    // CHECK-NEXT: async.yield
    async.yield
  }

  // CHECK: async.await %[[TOKEN]]
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.await %token : !async.token
  // CHECK: async.await %[[TOKEN_0]]
  // CHECK: async.drop_ref %[[TOKEN_0]] {count = 1 : i32}
  async.await %token_0 : !async.token

  // CHECK: return
  return
}

// CHECK-LABEL: @value_operand
func @value_operand() -> f32 {
  // CHECK: %[[TOKEN:.*]], %[[RESULTS:.*]] = async.execute
  %token, %results = async.execute -> !async.value<f32> {
    %0 = constant 0.0 : f32
    async.yield %0 : f32
  }

  // CHECK: async.add_ref %[[TOKEN]] {count = 1 : i32}
  // CHECK: async.add_ref %[[RESULTS]] {count = 1 : i32}
  // CHECK: %[[TOKEN_0:.*]] = async.execute
  %token_0 = async.execute[%token](%results as %arg0 : !async.value<f32>)  {
    // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
    // CHECK: async.drop_ref %[[RESULTS]] {count = 1 : i32}
    // CHECK: async.yield
    async.yield
  }

  // CHECK: async.await %[[TOKEN]]
  // CHECK: async.drop_ref %[[TOKEN]] {count = 1 : i32}
  async.await %token : !async.token

  // CHECK: async.await %[[TOKEN_0]]
  // CHECK: async.drop_ref %[[TOKEN_0]] {count = 1 : i32}
  async.await %token_0 : !async.token

  // CHECK: async.await %[[RESULTS]]
  // CHECK: async.drop_ref %[[RESULTS]] {count = 1 : i32}
  %0 = async.await %results : !async.value<f32>

  // CHECK: return
  return %0 : f32
}
