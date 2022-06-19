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
