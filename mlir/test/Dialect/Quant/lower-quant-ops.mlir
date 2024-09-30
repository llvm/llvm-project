// RUN: mlir-opt %s --lower-quant-ops --split-input-file | FileCheck %s

// CHECK-LABEL: @dcast_per_layer_scalar
// CHECK-SAME: %[[ARG_0:.*]]: !quant.uniform

// CHECK: %[[STORED_INT:.*]] = quant.scast %[[ARG_0]] : !quant.uniform<i8:f32, 2.000000e+00:10> to i8

// CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ZERO_POINT:.*]] = arith.constant 10 : i8
// CHECK: %[[STORED_FLOAT:.*]] = arith.sitofp %[[STORED_INT]] : i8 to f32
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32

// CHECK: %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK: %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE]] : f32
// CHECK: return %[[EXPRESSED]] : f32

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @dcast_per_layer_scalar(%arg0: !qalias) -> f32 {
  %0 = quant.dcast %arg0 : !qalias to f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @dcast_per_layer_scalar_unsigned
// CHECK-SAME: %[[ARG_0:.*]]: !quant.uniform

// CHECK: %[[STORED_INT:.*]] = quant.scast %[[ARG_0]] : !quant.uniform<u8:f32, 2.000000e+00:10> to i8

// CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ZERO_POINT:.*]] = arith.constant 10 : i8

// CHECK: %[[STORED_FLOAT:.*]] = arith.uitofp %[[STORED_INT]] : i8 to f32
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.uitofp %[[ZERO_POINT]] : i8 to f32

// CHECK: %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK: %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE]] : f32
// CHECK: return %[[EXPRESSED]] : f32

!qalias = !quant.uniform<u8:f32, 2.0:10>
func.func @dcast_per_layer_scalar_unsigned(%arg0: !qalias) -> f32 {
  %0 = quant.dcast %arg0 : !qalias to f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @dcast_per_layer_0d
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[STORED_INT:.*]] = quant.scast %[[ARG_0]] : tensor<!quant.uniform<i8:f32, 2.000000e+00:10>> to tensor<i8>

// CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ZERO_POINT:.*]] = arith.constant 10 : i8
// CHECK: %[[SCALE_TENSOR:.*]] = tensor.splat %[[SCALE]] : tensor<f32>
// CHECK: %[[STORED_FLOAT:.*]] = arith.sitofp %[[STORED_INT]] : tensor<i8> to tensor<f32>
// CHECK: %[[ZERO_POINT_TENSOR:.*]] = tensor.splat %[[ZERO_POINT]] : tensor<i8>
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_TENSOR]] : tensor<i8> to tensor<f32>

// CHECK: %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : tensor<f32>
// CHECK: %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE_TENSOR]] : tensor<f32>
// CHECK: return %[[EXPRESSED]] : tensor<f32>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @dcast_per_layer_0d(%arg0: tensor<!qalias>) -> tensor<f32> {
  %0 = quant.dcast %arg0 : tensor<!qalias> to tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @dcast_per_layer_ranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[STORED_INT:.*]] = quant.scast %[[ARG_0]] : tensor<3x?x5x!quant.uniform<i8:f32, 2.000000e+00:10>> to tensor<3x?x5xi8>
// CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ZERO_POINT:.*]] = arith.constant 10 : i8
// CHECK: %[[C_1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM_1:.*]] = tensor.dim %[[STORED_INT]], %[[C_1]] : tensor<3x?x5xi8>
// CHECK: %[[SCALE_TENSOR:.*]] = tensor.splat %[[SCALE]]{{\[}}%[[DIM_1]]] : tensor<3x?x5xf32>
// CHECK: %[[STORED_FLOAT:.*]] = arith.sitofp %[[STORED_INT]] : tensor<3x?x5xi8> to tensor<3x?x5xf32>
// CHECK: %[[ZERO_POINT_TENSOR:.*]] = tensor.splat %[[ZERO_POINT]]{{\[}}%[[DIM_1]]] : tensor<3x?x5xi8>
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_TENSOR]] : tensor<3x?x5xi8> to tensor<3x?x5xf32>

// CHECK: %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : tensor<3x?x5xf32>
// CHECK: %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE_TENSOR]] : tensor<3x?x5xf32>
// CHECK: return %[[EXPRESSED]] : tensor<3x?x5xf32>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @dcast_per_layer_ranked(%arg0: tensor<3x?x5x!qalias>) -> tensor<3x?x5xf32> {
  %0 = quant.dcast %arg0 : tensor<3x?x5x!qalias> to tensor<3x?x5xf32>
  return %0 : tensor<3x?x5xf32>
}

// -----

// CHECK-LABEL: @dcast_per_layer_unranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[STORED_INT:.*]] = quant.scast %[[ARG_0]] : tensor<*x!quant.uniform<i8:f32, 2.000000e+00:10>> to tensor<*xi8>
// CHECK: %[[INPUT_SHAPE:.*]] = shape.shape_of %[[STORED_INT]] : tensor<*xi8> -> tensor<?xindex>
// CHECK: %[[INPUT_SIZE:.*]] = shape.num_elements %[[INPUT_SHAPE]] : tensor<?xindex> -> index
// CHECK: %[[COLLAPSED_SHAPE:.*]] = tensor.from_elements %[[INPUT_SIZE]] : tensor<1xindex>
// CHECK: %[[STORED_COLLAPSED:.*]] = tensor.reshape %[[STORED_INT]](%[[COLLAPSED_SHAPE]]) : (tensor<*xi8>, tensor<1xindex>) -> tensor<?xi8>
// CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[ZERO_POINT:.*]] = arith.constant 10 : i8
// CHECK: %[[C_0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM_0:.*]] = tensor.dim %[[STORED_COLLAPSED]], %[[C_0]] : tensor<?xi8>
// CHECK: %[[SCALE_TENSOR:.*]] = tensor.splat %[[SCALE]]{{\[}}%[[DIM_0]]] : tensor<?xf32>
// CHECK: %[[STORED_FLOAT:.*]] = arith.sitofp %[[STORED_COLLAPSED]] : tensor<?xi8> to tensor<?xf32>
// CHECK: %[[ZERO_POINT_TENSOR:.*]] = tensor.splat %[[ZERO_POINT]]{{\[}}%[[DIM_0]]] : tensor<?xi8>
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_TENSOR]] : tensor<?xi8> to tensor<?xf32>

// CHECK: %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : tensor<?xf32>
// CHECK: %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE_TENSOR]] : tensor<?xf32>

// CHECK: %[[EXPRESSED_EXPANDED:.*]] = tensor.reshape %[[EXPRESSED]](%[[INPUT_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK: return %[[EXPRESSED_EXPANDED]] : tensor<*xf32>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @dcast_per_layer_unranked(%arg0: tensor<*x!qalias>) -> tensor<*xf32> {
  %0 = quant.dcast %arg0 : tensor<*x!qalias> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>

// CHECK-LABEL: @dcast_per_channel_ranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[STORED_TENSOR:.*]] = quant.scast %[[ARG_0]] : tensor<4x?x?x5x!quant.uniform<i8:f32:1, {2.000000e+00:10,3.000000e+00:20}>> to tensor<4x?x?x5xi8>

// CHECK: %[[SCALES:.*]] = arith.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
// CHECK: %[[ZERO_POINTS:.*]] = arith.constant dense<[10, 20]> : tensor<2xi8>
// CHECK: %[[C_1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM_1:.*]] = tensor.dim %[[STORED_TENSOR]], %[[C_1]] : tensor<4x?x?x5xi8>
// CHECK: %[[C_2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM_2:.*]] = tensor.dim %[[STORED_TENSOR]], %[[C_2]] : tensor<4x?x?x5xi8>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM_1]], %[[DIM_2]]) : tensor<4x?x?x5xf32>
// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[STORED_TENSOR]], %[[SCALES]], %[[ZERO_POINTS]] : tensor<4x?x?x5xi8>, tensor<2xf32>, tensor<2xi8>) outs(%[[INIT]] : tensor<4x?x?x5xf32>) {
// CHECK: ^bb0(%[[STORED_INT:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZERO_POINT:.*]]: i8, %[[OUT:.*]]: f32):
// CHECK:   %[[STORED_FLOAT:.*]] = arith.sitofp %[[STORED_INT]] : i8 to f32
// CHECK:   %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32
// CHECK:   %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK:   %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[EXPRESSED]] : f32
// CHECK: } -> tensor<4x?x?x5xf32>
// CHECK: return %[[GENERIC]] : tensor<4x?x?x5xf32>

!qalias = !quant.uniform<i8:f32:1, {2.0:10, 3.0:20}>
func.func @dcast_per_channel_ranked(%arg0: tensor<4x?x?x5x!qalias>) -> tensor<4x?x?x5xf32> {
  %0 = quant.dcast %arg0 : tensor<4x?x?x5x!qalias> to tensor<4x?x?x5xf32>
  return %0 : tensor<4x?x?x5xf32>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1)>

// CHECK-LABEL: @dcast_per_channel_unranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor

// CHECK: %[[STORED_TENSOR:.*]] = quant.scast %[[ARG_0]] : tensor<*x!quant.uniform<i8:f32:2, {2.000000e+00:10,3.000000e+00:20,4.000000e+00:30}>> to tensor<*xi8>
// CHECK: %[[SHAPE:.*]] = shape.shape_of %[[STORED_TENSOR]] : tensor<*xi8> -> tensor<?xindex>
// CHECK: %[[CHANNEL_AXIS:.*]] = arith.constant 2 : index
// CHECK: %[[CHANNEL_AXIS_NEXT:.*]] = arith.constant 3 : index
// CHECK: %[[SHAPE_LEFT:.*]], %[[DISCARDED_0:.*]] = "shape.split_at"(%[[SHAPE]], %[[CHANNEL_AXIS]]) : (tensor<?xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK: %[[SIZE_LEFT:.*]] = shape.num_elements %[[SHAPE_LEFT]] : tensor<?xindex> -> index
// CHECK: %[[DISCARDED_1:.*]], %[[SHAPE_RIGHT:.*]] = "shape.split_at"(%[[SHAPE]], %[[CHANNEL_AXIS_NEXT]]) : (tensor<?xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK: %[[SIZE_RIGHT:.*]] = shape.num_elements %[[SHAPE_RIGHT]] : tensor<?xindex> -> index

// CHECK: %[[NUM_CHANNELS:.*]] = arith.constant 3 : index
// CHECK: %[[COLLAPSED_SHAPE:.*]] = tensor.from_elements %[[SIZE_LEFT]], %[[NUM_CHANNELS]], %[[SIZE_RIGHT]] : tensor<3xindex>
// CHECK: %[[STORED_COLLAPSED:.*]] = tensor.reshape %[[STORED_TENSOR]](%[[COLLAPSED_SHAPE]]) : (tensor<*xi8>, tensor<3xindex>) -> tensor<?x3x?xi8>

// CHECK: %[[SCALES:.*]] = arith.constant dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf32>
// CHECK: %[[ZERO_POINTS:.*]] = arith.constant dense<[10, 20, 30]> : tensor<3xi8>
// CHECK: %[[C_0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM_0:.*]] = tensor.dim %[[STORED_COLLAPSED]], %[[C_0]] : tensor<?x3x?xi8>
// CHECK: %[[C_2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM_2:.*]] = tensor.dim %[[STORED_COLLAPSED]], %[[C_2]] : tensor<?x3x?xi8>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_2]]) : tensor<?x3x?xf32>
// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[STORED_COLLAPSED]], %[[SCALES]], %[[ZERO_POINTS]] : tensor<?x3x?xi8>, tensor<3xf32>, tensor<3xi8>) outs(%[[INIT]] : tensor<?x3x?xf32>) {
// CHECK: ^bb0(%[[STORED_INT:.*]]: i8, %[[SCALE:.*]]: f32, %[[ZERO_POINT:.*]]: i8, %[[OUT:.*]]: f32):
// CHECK:   %[[STORED_FLOAT:.*]] = arith.sitofp %[[STORED_INT]] : i8 to f32
// CHECK:   %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32
// CHECK:   %[[SCALED:.*]] = arith.subf %[[STORED_FLOAT]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK:   %[[EXPRESSED:.*]] = arith.mulf %[[SCALED]], %[[SCALE]] : f32
// CHECK:   linalg.yield %[[EXPRESSED]] : f32
// CHECK: } -> tensor<?x3x?xf32>

// CHECK: %[[EXPRESSED_EXPANDED:.*]] = tensor.reshape %[[GENERIC]](%[[SHAPE]]) : (tensor<?x3x?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK: return %[[EXPRESSED_EXPANDED]] : tensor<*xf32>

!qalias = !quant.uniform<i8:f32:2, {2.0:10, 3.0:20, 4.0:30}>
func.func @dcast_per_channel_unranked(%arg0: tensor<*x!qalias>) -> tensor<*xf32> {
  %0 = quant.dcast %arg0 : tensor<*x!qalias> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @qcast_per_layer_scalar
// CHECK-SAME: %[[ARG_0:.*]]: f32

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 10 : i8

// CHECK: %[[SCALED:.*]] = arith.divf %[[ARG_0]], %[[SCALE]] : f32
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32
// CHECK: %[[STORED:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK: %[[STORED_INT:.*]] = arith.fptosi %[[STORED]] : f32 to i8

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_INT]] : i8 to !quant.uniform<i8:f32, 2.000000e+00:10>
// CHECK: return %[[STORED_QUANT]] : !quant.uniform<i8:f32, 2.000000e+00:10>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @qcast_per_layer_scalar(%arg0: f32) -> !qalias {
  %0 = quant.qcast %arg0 : f32 to !qalias
  return %0 : !qalias
}

// -----

// CHECK-LABEL: @qcast_per_layer_scalar_bounds
// CHECK-SAME: %[[ARG_0:.*]]: f32

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 0 : i8

// CHECK: %[[SCALED:.*]] = arith.divf %[[ARG_0]], %[[SCALE]] : f32
// CHECK: %[[STORED_INT:.*]] = arith.fptosi %[[SCALED]] : f32 to i8

// CHECK-DAG: %[[C_NEG_5:.*]] = arith.constant -5 : i8
// CHECK-DAG: %[[C_10:.*]] = arith.constant 10 : i8
// CHECK: %[[STORED_CLAMPED_TEMP:.*]] = arith.maxsi %[[STORED_INT]], %[[C_NEG_5]] : i8
// CHECK: %[[STORED_CLAMPED:.*]] = arith.minsi %[[STORED_CLAMPED_TEMP]], %[[C_10]] : i8

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_CLAMPED]] : i8 to !quant.uniform<i8<-5:10>:f32, 2.000000e+00>
// CHECK: return %[[STORED_QUANT]] : !quant.uniform<i8<-5:10>:f32, 2.000000e+00>

!qalias = !quant.uniform<i8<-5:10>:f32, 2.0>
func.func @qcast_per_layer_scalar_bounds(%arg0: f32) -> !qalias {
  %0 = quant.qcast %arg0 : f32 to !qalias
  return %0 : !qalias
}

// -----

// CHECK-LABEL: @qcast_per_layer_scalar_unsigned_bounds
// CHECK-SAME: %[[ARG_0:.*]]: f32

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 0 : i8

// CHECK: %[[SCALED:.*]] = arith.divf %[[ARG_0]], %[[SCALE]] : f32
// CHECK: %[[STORED_INT:.*]] = arith.fptoui %[[SCALED]] : f32 to i8

// CHECK-DAG: %[[C_2:.*]] = arith.constant 2 : i8
// CHECK-DAG: %[[C_10:.*]] = arith.constant 10 : i8
// CHECK: %[[STORED_CLAMPED_TEMP:.*]] = arith.maxui %[[STORED_INT]], %[[C_2]] : i8
// CHECK: %[[STORED_CLAMPED:.*]] = arith.minui %[[STORED_CLAMPED_TEMP]], %[[C_10]] : i8

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_CLAMPED]] : i8 to !quant.uniform<u8<2:10>:f32, 2.000000e+00>
// CHECK: return %[[STORED_QUANT]] : !quant.uniform<u8<2:10>:f32, 2.000000e+00>

!qalias = !quant.uniform<u8<2:10>:f32, 2.0>
func.func @qcast_per_layer_scalar_unsigned_bounds(%arg0: f32) -> !qalias {
  %0 = quant.qcast %arg0 : f32 to !qalias
  return %0 : !qalias
}

// -----

// CHECK-LABEL: @qcast_per_layer_0d
// CHECK-SAME: %[[ARG_0:.*]]: tensor<f32>

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 10 : i8

// CHECK: %[[SCALE_TENSOR:.*]] = tensor.splat %[[SCALE]] : tensor<f32>
// CHECK: %[[SCALED:.*]] = arith.divf %[[ARG_0]], %[[SCALE_TENSOR]] : tensor<f32>

// CHECK: %[[ZERO_POINT_TENSOR:.*]] = tensor.splat %[[ZERO_POINT]] : tensor<i8>
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_TENSOR]] : tensor<i8> to tensor<f32>
// CHECK: %[[STORED_FLOAT:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : tensor<f32>
// CHECK: %[[STORED_INT:.*]] = arith.fptosi %[[STORED_FLOAT]] : tensor<f32> to tensor<i8>

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_INT]] : tensor<i8> to tensor<!quant.uniform<i8:f32, 2.000000e+00:10>>
// CHECK: return %[[STORED_QUANT]] : tensor<!quant.uniform<i8:f32, 2.000000e+00:10>>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @qcast_per_layer_0d(%arg0: tensor<f32>) -> tensor<!qalias> {
  %0 = quant.qcast %arg0 : tensor<f32> to tensor<!qalias>
  return %0 : tensor<!qalias>
}

// -----

// CHECK-LABEL: @qcast_per_layer_ranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor<3x?x5xf32>

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 10 : i8
// CHECK-DAG: %[[C_1:.*]] = arith.constant 1 : index

// CHECK: %[[DIM_1:.*]] = tensor.dim %[[ARG_0]], %[[C_1]] : tensor<3x?x5xf32>
// CHECK: %[[SCALE_TENSOR:.*]] = tensor.splat %[[SCALE]]{{\[}}%[[DIM_1]]] : tensor<3x?x5xf32>
// CHECK: %[[SCALED:.*]] = arith.divf %[[ARG_0]], %[[SCALE_TENSOR]] : tensor<3x?x5xf32>

// CHECK: %[[ZERO_POINT_TENSOR:.*]] = tensor.splat %[[ZERO_POINT]]{{\[}}%[[DIM_1]]] : tensor<3x?x5xi8>
// CHECK: %[[ZERO_POINT_TENSOR_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_TENSOR]] : tensor<3x?x5xi8> to tensor<3x?x5xf32>
// CHECK: %[[STORED:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_TENSOR_FLOAT]] : tensor<3x?x5xf32>
// CHECK: %[[STORED_INT:.*]] = arith.fptosi %[[STORED]] : tensor<3x?x5xf32> to tensor<3x?x5xi8>

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_INT]] : tensor<3x?x5xi8> to tensor<3x?x5x!quant.uniform<i8:f32, 2.000000e+00:10>>
// CHECK: return %[[STORED_QUANT]] : tensor<3x?x5x!quant.uniform<i8:f32, 2.000000e+00:10>>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @qcast_per_layer_ranked(%arg0: tensor<3x?x5xf32>) -> tensor<3x?x5x!qalias> {
  %0 = quant.qcast %arg0 : tensor<3x?x5xf32> to tensor<3x?x5x!qalias>
  return %0 : tensor<3x?x5x!qalias>
}

// -----

// CHECK-LABEL: @qcast_per_layer_ranked_bounds
// CHECK-SAME: %[[ARG_0:.*]]: tensor<3x5xf32>

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 10 : i8

// CHECK: %[[SCALE_SPLAT:.*]] = tensor.splat %[[SCALE]] : tensor<3x5xf32>
// CHECK: %[[SCALED:.*]] = arith.divf %[[ARG_0]], %[[SCALE_SPLAT]] : tensor<3x5xf32>

// CHECK: %[[ZERO_POINT_SPLAT:.*]] = tensor.splat %[[ZERO_POINT]] : tensor<3x5xi8>
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_SPLAT]] : tensor<3x5xi8> to tensor<3x5xf32>

// CHECK: %[[STORED_FLOAT:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : tensor<3x5xf32>
// CHECK: %[[STORED_INT:.*]] = arith.fptosi %[[STORED_FLOAT]] : tensor<3x5xf32> to tensor<3x5xi8>

// CHECK-DAG: %[[C_NEG_8:.*]] = arith.constant -8 : i8
// CHECK-DAG: %[[C_7:.*]] = arith.constant 7 : i8
// CHECK-DAG: %[[SPLAT_NEG_8:.*]] = tensor.splat %[[C_NEG_8]] : tensor<3x5xi8>
// CHECK-DAG: %[[SPLAT_7:.*]] = tensor.splat %[[C_7]] : tensor<3x5xi8>
// CHECK: %[[STORED_CLAMPED_TEMP:.*]] = arith.maxsi %[[STORED_INT]], %[[SPLAT_NEG_8]] : tensor<3x5xi8>
// CHECK: %[[STORED_CLAMPED:.*]] = arith.minsi %[[STORED_CLAMPED_TEMP]], %[[SPLAT_7]] : tensor<3x5xi8>

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_CLAMPED]] : tensor<3x5xi8> to tensor<3x5x!quant.uniform<i8<-8:7>:f32, 2.000000e+00:10>>
// CHECK: return %[[STORED_QUANT]] : tensor<3x5x!quant.uniform<i8<-8:7>:f32, 2.000000e+00:10>>

!qalias = !quant.uniform<i8<-8:7>:f32, 2.0:10>
func.func @qcast_per_layer_ranked_bounds(%arg0: tensor<3x5xf32>) -> tensor<3x5x!qalias> {
  %0 = quant.qcast %arg0 : tensor<3x5xf32> to tensor<3x5x!qalias>
  return %0 : tensor<3x5x!qalias>
}

// -----

// CHECK-LABEL: @qcast_per_layer_unranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*xf32>

// CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG_0]] : tensor<*xf32> -> tensor<?xindex>
// CHECK: %[[SIZE:.*]] = shape.num_elements %[[SHAPE]] : tensor<?xindex> -> index
// CHECK: %[[SIZE_TENSOR:.*]] = tensor.from_elements %[[SIZE]] : tensor<1xindex>
// CHECK: %[[RANKED_INPUT:.*]] = tensor.reshape %[[ARG_0]](%[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>

// CHECK-DAG: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG: %[[ZERO_POINT:.*]] = arith.constant 10 : i8
// CHECK-DAG: %[[C_0:.*]] = arith.constant 0 : index

// CHECK: %[[DIM_0:.*]] = tensor.dim %[[RANKED_INPUT]], %[[C_0]] : tensor<?xf32>
// CHECK: %[[SCALE_SPLAT:.*]] = tensor.splat %[[SCALE]]{{\[}}%[[DIM_0]]] : tensor<?xf32>
// CHECK: %[[SCALED:.*]] = arith.divf %[[RANKED_INPUT]], %[[SCALE_SPLAT]] : tensor<?xf32>

// CHECK: %[[ZERO_POINT_SPLAT:.*]] = tensor.splat %[[ZERO_POINT]]{{\[}}%[[DIM_0]]] : tensor<?xi8>
// CHECK: %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT_SPLAT]] : tensor<?xi8> to tensor<?xf32>
// CHECK: %[[STORED:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : tensor<?xf32>
// CHECK: %[[STORED_INT:.*]] = arith.fptosi %[[STORED]] : tensor<?xf32> to tensor<?xi8>

// CHECK: %[[STORED_UNRANKED:.*]] = tensor.reshape %[[STORED_INT]](%[[SHAPE]]) : (tensor<?xi8>, tensor<?xindex>) -> tensor<*xi8>
// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_UNRANKED]] : tensor<*xi8> to tensor<*x!quant.uniform<i8:f32, 2.000000e+00:10>>
// CHECK: return %[[STORED_QUANT]] : tensor<*x!quant.uniform<i8:f32, 2.000000e+00:10>>

!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @qcast_per_layer_unranked(%arg0: tensor<*xf32>) -> tensor<*x!qalias> {
  %0 = quant.qcast %arg0 : tensor<*xf32> to tensor<*x!qalias>
  return %0 : tensor<*x!qalias>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>

// CHECK-LABEL: @qcast_per_channel_ranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor<4x?x?x5xf32>

// CHECK: %[[SCALES:.*]] = arith.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
// CHECK: %[[ZERO_POINTS:.*]] = arith.constant dense<[10, 20]> : tensor<2xi8>

// CHECK-DAG: %[[C_1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM_1:.*]] = tensor.dim %[[ARG_0]], %[[C_1]] : tensor<4x?x?x5xf32>
// CHECK-DAG: %[[C_2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[DIM_2:.*]] = tensor.dim %[[ARG_0]], %[[C_2]] : tensor<4x?x?x5xf32>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM_1]], %[[DIM_2]]) : tensor<4x?x?x5xi8>

// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG_0]], %[[SCALES]], %[[ZERO_POINTS]] : tensor<4x?x?x5xf32>, tensor<2xf32>, tensor<2xi8>) outs(%[[INIT]] : tensor<4x?x?x5xi8>) {
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZERO_POINT:.*]]: i8, %[[OUT:.*]]: i8):
// CHECK:   %[[SCALED:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32
// CHECK:   %[[STORED_FLOAT:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK:   %[[STORED_INT:.*]] = arith.fptosi %[[STORED_FLOAT]] : f32 to i8
// CHECK:   linalg.yield %[[STORED_INT]] : i8
// CHECK: } -> tensor<4x?x?x5xi8>

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[GENERIC]] : tensor<4x?x?x5xi8> to tensor<4x?x?x5x!quant.uniform<i8:f32:1, {2.000000e+00:10,3.000000e+00:20}>>
// CHECK: return %[[STORED_QUANT]] : tensor<4x?x?x5x!quant.uniform<i8:f32:1, {2.000000e+00:10,3.000000e+00:20}>>

!qalias = !quant.uniform<i8:f32:1, {2.0:10, 3.0:20}>
func.func @qcast_per_channel_ranked(%arg0: tensor<4x?x?x5xf32>) -> tensor<4x?x?x5x!qalias> {
  %0 = quant.qcast %arg0 : tensor<4x?x?x5xf32> to tensor<4x?x?x5x!qalias>
  return %0 : tensor<4x?x?x5x!qalias>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1)>

// CHECK-LABEL: @qcast_per_channel_ranked_bounds
// CHECK-SAME: %[[ARG_0:.*]]: tensor<4x2x5xf32>

// CHECK: %[[SCALES:.*]] = arith.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
// CHECK: %[[ZERO_POINTS:.*]] = arith.constant dense<0> : tensor<2xi8>

// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x2x5xi8>
// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG_0]], %[[SCALES]], %[[ZERO_POINTS]] : tensor<4x2x5xf32>, tensor<2xf32>, tensor<2xi8>) outs(%[[INIT]] : tensor<4x2x5xi8>) {
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZERO_POINT:.*]]: i8, %[[OUT:.*]]: i8):
// CHECK:   %[[SCALED:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32
// CHECK:   %[[STORED_FLOAT:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK:   %[[STORED_INT:.*]] = arith.fptosi %[[STORED_FLOAT]] : f32 to i8
// CHECK:   %[[C_NEG_8:.*]] = arith.constant -8 : i8
// CHECK:   %[[C_7:.*]] = arith.constant 7 : i8
// CHECK:   %[[STORED_CLAMPED_TEMP:.*]] = arith.maxsi %[[STORED_INT]], %[[C_NEG_8]] : i8
// CHECK:   %[[STORED_CLAMPED:.*]] = arith.minsi %[[STORED_CLAMPED_TEMP]], %[[C_7]] : i8
// CHECK:   linalg.yield %[[STORED_CLAMPED]] : i8
// CHECK: } -> tensor<4x2x5xi8>

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[GENERIC]] : tensor<4x2x5xi8> to tensor<4x2x5x!quant.uniform<i8<-8:7>:f32:1, {2.000000e+00,3.000000e+00}>>
// CHECK: return %[[STORED_QUANT]] : tensor<4x2x5x!quant.uniform<i8<-8:7>:f32:1, {2.000000e+00,3.000000e+00}>>

!qalias = !quant.uniform<i8<-8:7>:f32:1, {2.0, 3.0}>
func.func @qcast_per_channel_ranked_bounds(%arg0: tensor<4x2x5xf32>) -> tensor<4x2x5x!qalias> {
  %0 = quant.qcast %arg0 : tensor<4x2x5xf32> to tensor<4x2x5x!qalias>
  return %0 : tensor<4x2x5x!qalias>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1)>

// CHECK-LABEL: @qcast_per_channel_unranked
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*xf32>

// CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG_0]] : tensor<*xf32> -> tensor<?xindex>
// CHECK: %[[CHANNEL_AXIS:.*]] = arith.constant 2 : index
// CHECK: %[[CHANNEL_AXIS_NEXT:.*]] = arith.constant 3 : index
// CHECK: %[[SHAPE_LEFT:.*]], %[[DISCARDED_0:.*]] = "shape.split_at"(%[[SHAPE]], %[[CHANNEL_AXIS]]) : (tensor<?xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK: %[[SIZE_LEFT:.*]] = shape.num_elements %[[SHAPE_LEFT]] : tensor<?xindex> -> index
// CHECK: %[[DISCARDED_1:.*]], %[[SHAPE_RIGHT:.*]] = "shape.split_at"(%[[SHAPE]], %[[CHANNEL_AXIS_NEXT]]) : (tensor<?xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK: %[[SIZE_RIGHT:.*]] = shape.num_elements %[[SHAPE_RIGHT]] : tensor<?xindex> -> index

// CHECK: %[[CHANNEL_AXIS_SIZE:.*]] = arith.constant 3 : index
// CHECK: %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[SIZE_LEFT]], %[[CHANNEL_AXIS_SIZE]], %[[SIZE_RIGHT]] : tensor<3xindex>
// CHECK: %[[FLAT_INPUT:.*]] = tensor.reshape %[[ARG_0]](%[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x3x?xf32>

// CHECK: %[[SCALES:.*]] = arith.constant dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf32>
// CHECK: %[[ZERO_POINTS:.*]] = arith.constant dense<[10, 20, 30]> : tensor<3xi8>

// CHECK: %[[C_0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM_0:.*]] = tensor.dim %[[FLAT_INPUT]], %[[C_0]] : tensor<?x3x?xf32>
// CHECK: %[[C_2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM_2:.*]] = tensor.dim %[[FLAT_INPUT]], %[[C_2]] : tensor<?x3x?xf32>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM_0]], %[[DIM_2]]) : tensor<?x3x?xi8>

// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[FLAT_INPUT]], %[[SCALES]], %[[ZERO_POINTS]] : tensor<?x3x?xf32>, tensor<3xf32>, tensor<3xi8>) outs(%[[INIT]] : tensor<?x3x?xi8>) {
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[SCALE:.*]]: f32, %[[ZERO_POINT:.*]]: i8, %[[OUT:.*]]: i8):
// CHECK:   %[[SCALED:.*]] = arith.divf %[[IN]], %[[SCALE]] : f32
// CHECK:   %[[ZERO_POINT_FLOAT:.*]] = arith.sitofp %[[ZERO_POINT]] : i8 to f32
// CHECK:   %[[STORED_FLOAT:.*]] = arith.addf %[[SCALED]], %[[ZERO_POINT_FLOAT]] : f32
// CHECK:   %[[STORED_INT:.*]] = arith.fptosi %[[STORED_FLOAT]] : f32 to i8
// CHECK:   linalg.yield %[[STORED_INT]] : i8
// CHECK: } -> tensor<?x3x?xi8>

// CHECK: %[[STORED_UNRANKED:.*]] = tensor.reshape %[[GENERIC]](%[[SHAPE]]) : (tensor<?x3x?xi8>, tensor<?xindex>) -> tensor<*xi8>
// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_UNRANKED]] : tensor<*xi8> to tensor<*x!quant.uniform<i8:f32:2, {2.000000e+00:10,3.000000e+00:20,4.000000e+00:30}>>
// CHECK: return %[[STORED_QUANT]] : tensor<*x!quant.uniform<i8:f32:2, {2.000000e+00:10,3.000000e+00:20,4.000000e+00:30}>>

!qalias = !quant.uniform<i8:f32:2, {2.0:10, 3.0:20, 4.0:30}>
func.func @qcast_per_channel_unranked(%arg0: tensor<*xf32>) -> tensor<*x!qalias> {
  %0 = quant.qcast %arg0 : tensor<*xf32> to tensor<*x!qalias>
  return %0 : tensor<*x!qalias>
}

