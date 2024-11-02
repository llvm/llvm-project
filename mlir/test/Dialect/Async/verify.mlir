// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// FileCheck test must have at least one CHECK statement.
// CHECK-LABEL: @no_op
func.func @no_op(%arg0: !async.token) {
  return
}

// -----

func.func @wrong_async_await_arg_type(%arg0: f32) {
  // expected-error @+1 {{'async.await' op operand #0 must be async value type or async token type, but got 'f32'}}
  async.await %arg0 : f32
}

// -----

func.func @wrong_async_await_result_type(%arg0: !async.value<f32>) {
  // expected-error @+1 {{'async.await' op result type 'f64' does not match async value type 'f32'}}
  %0 = "async.await"(%arg0): (!async.value<f32>) -> f64
}


// -----
// expected-error @+1 {{'async.func' op result is expected to be at least of size 1, but got 0}}
async.func @wrong_async_func_void_result_type(%arg0: f32) {
  return
}


// -----
// expected-error @+1 {{'async.func' op result type must be async value type or async token type, but got 'f32'}}
async.func @wrong_async_func_result_type(%arg0: f32) -> f32 {
  return %arg0 : f32
}

// -----
// expected-error @+1 {{'async.func' op  results' (optional) async token type is expected to appear as the 1st return value, but got 2}}
async.func @wrong_async_func_token_type_placement(%arg0: f32) -> (!async.value<f32>, !async.token) {
  return %arg0 : f32
}

// -----
async.func @wrong_async_func_return_type(%arg0: f32) -> (!async.token, !async.value<i32>) {
  // expected-error @+1 {{'async.return' op operand types do not match the types returned from the parent FuncOp}}
  return %arg0 : f32
}
