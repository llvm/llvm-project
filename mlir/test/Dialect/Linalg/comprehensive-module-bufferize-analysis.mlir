// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize=test-analysis-only -split-input-file | FileCheck %s

func private @foo(tensor<64xf32>)

// CHECK-LABEL: dependence_through_call
func @dependence_through_call(%I : tensor<64xf32> {linalg.inplaceable = true}) {
  %f1 = constant 1.000000e+00 : f32
  %f2 = constant 2.000000e+00 : f32

  // 2. %B already bufferizes inplace, %A would alias and have a different
  // value. The calls to `foo` are determined to read conservatively, so %A
  // cannot bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %A = linalg.fill(%f1, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  // 1. Bufferizes inplace: no alias to %A is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %B = linalg.fill(%f2, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  call @foo(%A) : (tensor<64xf32>) -> ()
  call @foo(%B) : (tensor<64xf32>) -> ()

  return
}

// -----

func private @foo(tensor<64xf32>)

func private @bar(%A : tensor<64xf32>) {
  call @foo(%A) : (tensor<64xf32>) -> ()
  return
}

func @read_dependence_through_scf_and_call(
    %I : tensor<64xf32> {linalg.inplaceable = true},
    %I2 : tensor<64xf32> {linalg.inplaceable = true}) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %f1 = constant 1.000000e+00 : f32
  %f2 = constant 2.000000e+00 : f32

  // 5. %B bufferizes inplace, %A would alias and have a different value.
  // The calls to `foo` are determined to read conservatively, so %A cannot
  // bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %A = linalg.fill(%f1, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  // 4. Bufferizes inplace: no alias to %A is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %B = linalg.fill(%f2, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  // 3. Does not read or write, bufferizes inplace.
  //     CHECK: scf.for
  //     CHECK: {__inplace_results_attr__ = ["true", "true"]}
  %r:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%0 = %A, %1 = %B)
    -> (tensor<64xf32>, tensor<64xf32>)
  {
    scf.yield %0, %1 : tensor<64xf32>, tensor<64xf32>
  }
  call @foo(%r#0) : (tensor<64xf32>) -> ()
  call @foo(%r#1) : (tensor<64xf32>) -> ()

  // 2. %B2 already bufferizes inplace, %A2 would alias and have a different
  // value. The calls to `foo` are determined to read conservatively, so %A2
  // cannot bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %A2 = linalg.fill(%f1, %I2) : f32, tensor<64xf32> -> tensor<64xf32>

  // 1. Bufferizes inplace: no alias to %A2 is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %B2 = linalg.fill(%f2, %I2) : f32, tensor<64xf32> -> tensor<64xf32>

  call @bar(%A2) : (tensor<64xf32>) -> ()
  call @bar(%B2) : (tensor<64xf32>) -> ()
  return
}
