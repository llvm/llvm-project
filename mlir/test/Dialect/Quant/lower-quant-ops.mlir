// RUN: mlir-opt %s --lower-quant-ops --split-input-file | FileCheck %s

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
// CHECK: %[[STORED_FLOAT:.*]] = arith.fptosi %[[STORED]] : tensor<3x?x5xf32> to tensor<3x?x5xi8>

// CHECK: %[[STORED_QUANT:.*]] = quant.scast %[[STORED_FLOAT]] : tensor<3x?x5xi8> to tensor<3x?x5x!quant.uniform<i8:f32, 2.000000e+00:10>>
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

