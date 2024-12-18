// RUN: mlir-opt %s -split-input-file -verify-diagnostics

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_scalar(%arg0: !qalias) {
  %0 = quant.dcast %arg0 : !qalias to f32
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_ranked(%arg0: tensor<2x?x4x!qalias>) {
  %0 = quant.dcast %arg0 : tensor<2x?x4x!qalias> to tensor<2x?x4xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @dcast_unranked(%arg0: tensor<*x!qalias>) {
  %0 = quant.dcast %arg0 : tensor<*x!qalias> to tensor<*xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @dcast_per_axis_static(%arg0: tensor<1x2x3x!qalias>) {
  %0 = quant.dcast %arg0 : tensor<1x2x3x!qalias> to tensor<1x2x3xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @dcast_per_axis_dynamic(%arg0: tensor<?x?x?x!qalias>) {
  %0 = quant.dcast %arg0 : tensor<?x?x?x!qalias> to tensor<?x?x?xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @dcast_per_axis_unranked(%arg0: tensor<*x!qalias>) {
  %0 = quant.dcast %arg0 : tensor<*x!qalias> to tensor<*xf32>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_scalar(%arg0: f32) {
  %0 = quant.qcast %arg0 : f32 to !qalias
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_ranked(%arg0: tensor<2x?x4xf32>) {
  %0 = quant.qcast %arg0 : tensor<2x?x4xf32> to tensor<2x?x4x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @qcast_unranked(%arg0: tensor<*xf32>) {
  %0 = quant.qcast %arg0 : tensor<*xf32> to tensor<*x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @qcast_per_axis_static(%arg0: tensor<1x2x3xf32>) {
  %0 = quant.qcast %arg0 : tensor<1x2x3xf32> to tensor<1x2x3x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @qcast_per_axis_dynamic(%arg0: tensor<?x?x?xf32>) {
  %0 = quant.qcast %arg0 : tensor<?x?x?xf32> to tensor<?x?x?x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @qcast_per_axis_unranked(%arg0: tensor<*xf32>) {
  %0 = quant.qcast %arg0 : tensor<*xf32> to tensor<*x!qalias>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_scalar(%arg0: i8) {
  %0 = quant.scast %arg0 : i8 to !qalias
  %1 = quant.scast %0 : !qalias to i8
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_ranked(%arg0: tensor<2x?x4xi8>) {
  %0 = quant.scast %arg0 : tensor<2x?x4xi8> to tensor<2x?x4x!qalias>
  %1 = quant.scast %0 : tensor<2x?x4x!qalias> to tensor<2x?x4xi8>
  return
}

// -----

!qalias = !quant.uniform<i8:f32, 1.0>
func.func @scast_unranked(%arg0: tensor<*xi8>) {
  %0 = quant.scast %arg0 : tensor<*xi8> to tensor<*x!qalias>
  %1 = quant.scast %0 : tensor<*x!qalias> to tensor<*xi8>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @scast_per_axis_static(%arg0: tensor<1x2x3xi8>) {
  %0 = quant.scast %arg0 : tensor<1x2x3xi8> to tensor<1x2x3x!qalias>
  %1 = quant.scast %0 : tensor<1x2x3x!qalias> to tensor<1x2x3xi8>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @scast_per_axis_dynamic(%arg0: tensor<?x?x?xi8>) {
  %0 = quant.scast %arg0 : tensor<?x?x?xi8> to tensor<?x?x?x!qalias>
  %1 = quant.scast %0 : tensor<?x?x?x!qalias> to tensor<?x?x?xi8>
  return
}

// -----

!qalias = !quant.uniform<i8:f32:2, {1.0, 2.0, 3.0}>
func.func @scast_per_axis_unranked(%arg0: tensor<*xi8>) {
  %0 = quant.scast %arg0 : tensor<*xi8> to tensor<*x!qalias>
  %1 = quant.scast %0 : tensor<*x!qalias> to tensor<*xi8>
  return
}


