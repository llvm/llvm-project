// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @dcast_invalid_input(%arg0: f32) {
  // expected-error@+1 {{operand #0 must be scalar or tensor of quantized type}}
  %0 = quant.dcast %arg0 : f32 to f32
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_invalid_result(%arg0: !qalias) {
  // expected-error@+1 {{result #0 must be scalar or tensor of floating-point}}
  %0 = quant.dcast %arg0 : !qalias to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_mismatch_scalar_tensor(%arg0: !qalias) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.dcast %arg0 : !qalias to tensor<f32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_mismatch_ranked_unranked_tensor(%arg0: tensor<!qalias>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.dcast %arg0 : tensor<!qalias> to tensor<*xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_mismatch_static_dynamic_tensor(%arg0: tensor<2x3x!qalias>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.dcast %arg0 : tensor<2x3x!qalias> to tensor<?x3xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_float_type_mismatch(%arg0: !qalias) {
  // expected-error@+1 {{expressed type in quantized type expected to match float type}}
  %0 = quant.dcast %arg0 : !qalias to f64
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0}>
func.func @dcast_per_axis_scalar(%arg0: !qalias) {
  // expected-error@+1 {{scalar types may not use per-axis quantization}}
  %0 = quant.dcast %arg0 : !qalias to f32
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0}>
func.func @dcast_per_axis_invalid_rank(%arg0: tensor<2x3x!qalias>) {
  // expected-error@+1 {{quantized dimension must be less than tensor rank}}
  %0 = quant.dcast %arg0 : tensor<2x3x!qalias> to tensor<2x3xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @dcast_per_axis_invalid_rank(%arg0: tensor<2x3x4x!qalias>) {
  // expected-error@+1 {{quantized dimension size does not match number of scales}}
  %0 = quant.dcast %arg0 : tensor<2x3x4x!qalias> to tensor<2x3x4xf32>
  return
}

// -----

func.func @qcast_invalid_input(%arg0: f32) {
  // expected-error@+1 {{result #0 must be scalar or tensor of quantized type}}
  %0 = quant.qcast %arg0 : f32 to f32
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_invalid_result(%arg0: !qalias) {
  // expected-error@+1 {{operand #0 must be scalar or tensor of floating-point}}
  %0 = quant.qcast %arg0 : !qalias to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_mismatch_scalar_tensor(%arg0: tensor<f32>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.qcast %arg0 : tensor<f32> to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_mismatch_ranked_unranked_tensor(%arg0: tensor<f32>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.qcast %arg0 : tensor<f32> to tensor<*x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_mismatch_static_dynamic_tensor(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.qcast %arg0 : tensor<2x3xf32> to tensor<?x3x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_float_type_mismatch(%arg0: f64) {
  // expected-error@+1 {{expressed type in quantized type expected to match float type}}
  %0 = quant.qcast %arg0 : f64 to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0}>
func.func @qcast_per_axis_scalar(%arg0: f32) {
  // expected-error@+1 {{scalar types may not use per-axis quantization}}
  %0 = quant.qcast %arg0 : f32 to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0}>
func.func @qcast_per_axis_invalid_rank(%arg0: tensor<2x3xf32>) {
  // expected-error@+1 {{quantized dimension must be less than tensor rank}}
  %0 = quant.qcast %arg0 : tensor<2x3xf32> to tensor<2x3x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @qcast_per_axis_invalid_rank(%arg0: tensor<2x3x4xf32>) {
  // expected-error@+1 {{quantized dimension size does not match number of scales}}
  %0 = quant.qcast %arg0 : tensor<2x3x4xf32> to tensor<2x3x4x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_invalid_input(%arg0: si32) {
  // expected-error@+1 {{operand #0 must be scalar or tensor of signless integer or quantized type}}
  %0 = quant.scast %arg0 : si32 to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_invalid_result(%arg0: !qalias) {
  // expected-error@+1 {{result #0 must be scalar or tensor of signless integer or quantized type}}
  %0 = quant.scast %arg0 : !qalias to si32
  return
}

// -----

func.func @scast_both_integers(%arg0: i8) {
  // expected-error@+1 {{input must be integer and result must be quantized, or vice versa}}
  %0 = quant.scast %arg0 : i8 to i8
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_both_quantized(%arg0: !qalias) {
  // expected-error@+1 {{input must be integer and result must be quantized, or vice versa}}
  %0 = quant.scast %arg0 : !qalias to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_mismatch_scalar_tensor(%arg0: tensor<i8>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.scast %arg0 : tensor<i8> to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_mismatch_ranked_unranked_tensor(%arg0: tensor<i8>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.scast %arg0 : tensor<i8> to tensor<*x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_mismatch_static_dynamic_tensor(%arg0: tensor<2x3xi8>) {
  // expected-error@+1 {{input and result are both scalars or both tensors with matching shape}}
  %0 = quant.scast %arg0 : tensor<2x3xi8> to tensor<?x3x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_integer_type_mismatch(%arg0: i32) {
  // expected-error@+1 {{storage type in quantized type expected to match integer type}}
  %0 = quant.scast %arg0 : i32 to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0}>
func.func @scast_per_axis_scalar(%arg0: i8) {
  // expected-error@+1 {{scalar types may not use per-axis quantization}}
  %0 = quant.scast %arg0 : i8 to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0}>
func.func @scast_per_axis_invalid_rank(%arg0: tensor<2x3xi8>) {
  // expected-error@+1 {{quantized dimension must be less than tensor rank}}
  %0 = quant.scast %arg0 : tensor<2x3xi8> to tensor<2x3x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @scast_per_axis_invalid_rank(%arg0: tensor<2x3x4xi8>) {
  // expected-error@+1 {{quantized dimension size does not match number of scales}}
  %0 = quant.scast %arg0 : tensor<2x3x4xi8> to tensor<2x3x4x!qalias>
  return
}

