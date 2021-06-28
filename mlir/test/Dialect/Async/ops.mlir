// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: @identity_token
func @identity_token(%arg0: !async.token) -> !async.token {
  // CHECK: return %arg0 : !async.token
  return %arg0 : !async.token
}

// CHECK-LABEL: @identity_value
func @identity_value(%arg0 : !async.value<f32>) -> !async.value<f32> {
  // CHECK: return %arg0 : !async.value<f32>
  return %arg0 : !async.value<f32>
}

// CHECK-LABEL: @empty_async_execute
func @empty_async_execute() -> !async.token {
  // CHECK: async.execute
  %token = async.execute {
    async.yield
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @return_async_value
func @return_async_value() -> !async.value<f32> {
  // CHECK: async.execute -> !async.value<f32>
  %token, %results = async.execute -> !async.value<f32> {
    %cst = constant 1.000000e+00 : f32
    async.yield %cst : f32
  }

  // CHECK: return %results : !async.value<f32>
  return %results : !async.value<f32>
}

// CHECK-LABEL: @return_captured_value
func @return_captured_value() -> !async.token {
  %cst = constant 1.000000e+00 : f32
  // CHECK: async.execute -> !async.value<f32>
  %token, %results = async.execute -> !async.value<f32> {
    async.yield %cst : f32
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @return_async_values
func @return_async_values() -> (!async.value<f32>, !async.value<f32>) {
  %token, %results:2 = async.execute -> (!async.value<f32>, !async.value<f32>) {
    %cst1 = constant 1.000000e+00 : f32
    %cst2 = constant 2.000000e+00 : f32
    async.yield %cst1, %cst2 : f32, f32
  }

  // CHECK: return %results#0, %results#1 : !async.value<f32>, !async.value<f32>
  return %results#0, %results#1 : !async.value<f32>, !async.value<f32>
}

// CHECK-LABEL: @async_token_dependencies
func @async_token_dependencies(%arg0: !async.token) -> !async.token {
  // CHECK: async.execute [%arg0]
  %token = async.execute [%arg0] {
    async.yield
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @async_value_operands
func @async_value_operands(%arg0: !async.value<f32>) -> !async.token {
  // CHECK: async.execute (%arg0 as %arg1: !async.value<f32>) -> !async.value<f32>
  %token, %results = async.execute (%arg0 as %arg1: !async.value<f32>) -> !async.value<f32> {
    async.yield %arg1 : f32
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @async_token_and_value_operands
func @async_token_and_value_operands(%arg0: !async.token, %arg1: !async.value<f32>) -> !async.token {
  // CHECK: async.execute [%arg0] (%arg1 as %arg2: !async.value<f32>) -> !async.value<f32>
  %token, %results = async.execute [%arg0] (%arg1 as %arg2: !async.value<f32>) -> !async.value<f32> {
    async.yield %arg2 : f32
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @empty_tokens_or_values_operands
func @empty_tokens_or_values_operands() {
  // CHECK: async.execute {
  %token0 = async.execute [] () -> () { async.yield }
  // CHECK: async.execute {
  %token1 = async.execute () -> () { async.yield }
  // CHECK: async.execute {
  %token2 = async.execute -> () { async.yield }
  // CHECK: async.execute {
  %token3 = async.execute () { async.yield }
  // CHECK: async.execute {
  %token4 = async.execute [] { async.yield }
  return
}

// CHECK-LABEL: @await_token
func @await_token(%arg0: !async.token) {
  // CHECK: async.await %arg0
  async.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func @await_value(%arg0: !async.value<f32>) -> f32 {
  // CHECK: async.await %arg0
  %0 = async.await %arg0 : !async.value<f32>
  return %0 : f32
}

// CHECK-LABEL: @create_group_and_await_all
func @create_group_and_await_all(%arg0: !async.token,
                                 %arg1: !async.value<f32>) -> index {
  %c = constant 2 : index
  %0 = async.create_group %c : !async.group

  // CHECK: async.add_to_group %arg0
  // CHECK: async.add_to_group %arg1
  %1 = async.add_to_group %arg0, %0 : !async.token
  %2 = async.add_to_group %arg1, %0 : !async.value<f32>
  async.await_all %0

  %3 = addi %1, %2 : index
  return %3 : index
}
