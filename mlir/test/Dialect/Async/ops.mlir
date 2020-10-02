// RUN: mlir-opt  %s | FileCheck %s

// CHECK-LABEL: @identity_token
func @identity_token(%arg0 : !async.token) -> !async.token {
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
  %done = async.execute {
    async.yield
  } : !async.token

  // CHECK: return %done : !async.token
  return %done : !async.token
}

// CHECK-LABEL: @return_async_value
func @return_async_value() -> !async.value<f32> {
  %done, %values = async.execute {
    %cst = constant 1.000000e+00 : f32
    async.yield %cst : f32
  } : !async.token, !async.value<f32>

  // CHECK: return %values : !async.value<f32>
  return %values : !async.value<f32>
}

// CHECK-LABEL: @return_async_values
func @return_async_values() -> (!async.value<f32>, !async.value<f32>) {
  %done, %values:2 = async.execute {
    %cst1 = constant 1.000000e+00 : f32
    %cst2 = constant 2.000000e+00 : f32
    async.yield %cst1, %cst2 : f32, f32
  } : !async.token, !async.value<f32>, !async.value<f32>

  // CHECK: return %values#0, %values#1 : !async.value<f32>, !async.value<f32>
  return %values#0, %values#1 : !async.value<f32>, !async.value<f32>
}
