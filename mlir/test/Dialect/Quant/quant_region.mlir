// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @source
func.func @source(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = "quant.region"(%arg0, %arg1, %arg2) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>, %12: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %14 = "bar"(%13, %12) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [f32, f32, f32], output_specs = [f32], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @annotated
func.func @annotated(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = "quant.region"(%arg0, %arg1, %arg2) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>, %12: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %14 = "bar"(%13, %12) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0>, !quant.uniform<i8:f32, 2.0>, f32],
      output_specs = [!quant.uniform<i8:f32, 4.0>], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @quantized
func.func @quantized(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = "quant.region"(%arg0, %arg1, %arg2) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>, %12: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %14 = "bar"(%13, %12) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0>, !quant.uniform<i8:f32, 2.0>, !quant.uniform<i32:f32, 2.0>],
      output_specs = [!quant.uniform<i8:f32, 4.0>], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----

func.func @unmatched_quantize(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  // @expected-error @+1 {{'quant.region' op has incompatible specification !quant.uniform<i32:f16, 3.000000e+00> and input type 'tensor<4xf32>'}}
  %0 = "quant.region"(%arg0, %arg1, %arg2) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>, %12: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %14 = "bar"(%13, %12) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0>, !quant.uniform<i8:f32, 2.0>, !quant.uniform<i32:f16, 3.0>],
      output_specs = [!quant.uniform<i8:f32, 4.0>], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----

func.func @unmatched_primitive(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  // @expected-error @+1 {{'quant.region' op has incompatible specification i32 and input type 'tensor<4xf32>'}}
  %0 = "quant.region"(%arg0, %arg1, %arg2) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>, %12: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %14 = "bar"(%13, %12) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0>, !quant.uniform<i8:f32, 2.0>, i32],
      output_specs = [!quant.uniform<i8:f32, 4.0>], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----

func.func @unmatched_number(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  // @expected-error @+1 {{'quant.region' op has unmatched operands/results number and spec attributes number}}
  %0 = "quant.region"(%arg0, %arg1, %arg2) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>, %12: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %14 = "bar"(%13, %12) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0>, !quant.uniform<i8:f32, 2.0>],
      output_specs = [!quant.uniform<i8:f32, 4.0>], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

// -----

func.func @isolated(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  // @expected-note @+1 {{required by region isolation constraints}}
  %0 = "quant.region"(%arg0, %arg1) ({
    ^bb0(%10: tensor<4xf32>, %11: tensor<4xf32>):
      %13 = "foo"(%10, %11) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      // @expected-error @+1 {{'bar' op using value defined outside the region}}
      %14 = "bar"(%13, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "quant.return"(%14) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0>, !quant.uniform<i8:f32, 2.0>],
      output_specs = [!quant.uniform<i8:f32, 4.0>], logical_kernel = "xyz"}
    : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %0 : tensor<4xf32>
}

