// RUN: mlir-opt %s \
// RUN:     -one-shot-bufferize="allow-return-allocs create-deallocs=0" \
// RUN:     -split-input-file | \
// RUN: FileCheck %s --dump-input=always

// A regression test to check that different before and after argument types are
// bufferized successfully.
func.func @different_before_after_args() -> tensor<f32> {
  %true = arith.constant true
  %cst = arith.constant dense<0.0> : tensor<f32>
  %0 = scf.while (%arg4 = %true) : (i1) -> (tensor<f32>) {
    scf.condition(%true) %cst : tensor<f32>
  } do {
  ^bb0(%arg4: tensor<f32>):
    scf.yield %true : i1
  }
  return %0 : tensor<f32>
}

// CHECK-LABEL: @different_before_after_args