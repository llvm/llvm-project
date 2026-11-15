// RUN: mlir-opt --split-input-file --verify-each --tosa-reduce-reshapes %s | FileCheck %s

// CHECK-LABEL: @propagate_through_chain
// CHECK-DAG: %[[BIAS:.*]] = tosa.const values(dense<1.000000e+00> : tensor<1x1x1xf32>) : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[INPUT:.*]] = tosa.reshape %arg0{{.*}} -> tensor<2x3x4xf32>
// CHECK: %[[ADD:.*]] = tosa.add %[[INPUT]], %[[BIAS]] : (tensor<2x3x4xf32>, tensor<1x1x1xf32>) -> tensor<2x3x4xf32>
// CHECK: %[[ABS:.*]] = tosa.abs %[[ADD]] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK: %[[CAST:.*]] = tosa.cast %[[ABS]]{{.*}} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf16>
// CHECK: return %[[CAST]]
func.func @propagate_through_chain(%arg0: tensor<24xf32>) -> tensor<2x3x4xf16> {
  %bias = "tosa.const"() <{values = dense<1.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %add = tosa.add %arg0, %bias : (tensor<24xf32>, tensor<1xf32>) -> tensor<24xf32>
  %abs = tosa.abs %add : (tensor<24xf32>) -> tensor<24xf32>
  %cast = tosa.cast %abs : (tensor<24xf32>) -> tensor<24xf16>
  %shape = tosa.const_shape values(dense<[2, 3, 4]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %cast, %shape : (tensor<24xf16>, !tosa.shape<3>) -> tensor<2x3x4xf16>
  return %reshape : tensor<2x3x4xf16>
}

// -----

// CHECK-LABEL: @propagate_through_cast
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0{{.*}} -> tensor<2x3x4xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]{{.*}} : (tensor<2x3x4xf16>) -> tensor<2x3x4xf32>
// CHECK: return %[[CAST]]
func.func @propagate_through_cast(%arg0: tensor<24xf16>) -> tensor<2x3x4xf32> {
  %cast = tosa.cast %arg0 : (tensor<24xf16>) -> tensor<24xf32>
  %shape = tosa.const_shape values(dense<[2, 3, 4]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %cast, %shape : (tensor<24xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  return %reshape : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: @propagate_scalar_through_cast
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0{{.*}} -> tensor<1x1xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]{{.*}} : (tensor<1x1xf16>) -> tensor<1x1xf32>
// CHECK: return %[[CAST]]
func.func @propagate_scalar_through_cast(%arg0: tensor<f16>) -> tensor<1x1xf32> {
  %cast = tosa.cast %arg0 : (tensor<f16>) -> tensor<f32>
  %shape = tosa.const_shape values(dense<[1, 1]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %reshape = tosa.reshape %cast, %shape : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
  return %reshape : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: @keep_scalar_collapse
// CHECK: %[[CAST:.*]] = tosa.cast %arg0{{.*}} : (tensor<1x1xf16>) -> tensor<1x1xf32>
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[CAST]]{{.*}} -> tensor<f32>
// CHECK: return %[[RESHAPE]]
func.func @keep_scalar_collapse(%arg0: tensor<1x1xf16>) -> tensor<f32> {
  %cast = tosa.cast %arg0 : (tensor<1x1xf16>) -> tensor<1x1xf32>
  %shape = tosa.const_shape values(dense<> : tensor<0xindex>) : () -> !tosa.shape<0>
  %reshape = tosa.reshape %cast, %shape : (tensor<1x1xf32>, !tosa.shape<0>) -> tensor<f32>
  return %reshape : tensor<f32>
}

// -----

// CHECK-LABEL: @propagate_through_integer_cast
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg0{{.*}} -> tensor<2x3x4xi8>
// CHECK: %[[CAST:.*]] = tosa.cast %[[RESHAPE]]{{.*}} : (tensor<2x3x4xi8>) -> tensor<2x3x4xi32>
// CHECK: return %[[CAST]]
func.func @propagate_through_integer_cast(%arg0: tensor<24xi8>) -> tensor<2x3x4xi32> {
  %cast = tosa.cast %arg0 : (tensor<24xi8>) -> tensor<24xi32>
  %shape = tosa.const_shape values(dense<[2, 3, 4]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %cast, %shape : (tensor<24xi32>, !tosa.shape<3>) -> tensor<2x3x4xi32>
  return %reshape : tensor<2x3x4xi32>
}

// -----

// CHECK-LABEL: @keep_cast_from_block_scaled
// CHECK: %[[CAST:.*]] = tosa.cast %arg0
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[CAST]]{{.*}} -> tensor<32x2x16xf32>
// CHECK: return %[[RESHAPE]]
func.func @keep_cast_from_block_scaled(%arg0: tensor<32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<32x2x16xf32> {
  %cast = tosa.cast %arg0 : (tensor<32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<32x32xf32>
  %shape = tosa.const_shape values(dense<[32, 2, 16]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %cast, %shape : (tensor<32x32xf32>, !tosa.shape<3>) -> tensor<32x2x16xf32>
  return %reshape : tensor<32x2x16xf32>
}

// -----

// CHECK-LABEL: @keep_cast_to_block_scaled
// CHECK: %[[CAST:.*]] = tosa.cast %arg0
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[CAST]]{{.*}} -> tensor<32x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// CHECK: return %[[RESHAPE]]
func.func @keep_cast_to_block_scaled(%arg0: tensor<32x32xf32>) -> tensor<32x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>> {
  %cast = tosa.cast %arg0 : (tensor<32x32xf32>) -> tensor<32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
  %shape = tosa.const_shape values(dense<[32, 1, 32]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %cast, %shape : (tensor<32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>, !tosa.shape<3>) -> tensor<32x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
  return %reshape : tensor<32x1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
}

// -----

// CHECK-LABEL: @keep_two_non_foldable_operands
// CHECK: %[[ADD:.*]] = tosa.add %arg0, %arg1 : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[ADD]]{{.*}} -> tensor<2x3x4xf32>
// CHECK: return %[[RESHAPE]]
func.func @keep_two_non_foldable_operands(%arg0: tensor<24xf32>, %arg1: tensor<24xf32>) -> tensor<2x3x4xf32> {
  %add = tosa.add %arg0, %arg1 : (tensor<24xf32>, tensor<24xf32>) -> tensor<24xf32>
  %shape = tosa.const_shape values(dense<[2, 3, 4]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %add, %shape : (tensor<24xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  return %reshape : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: @keep_shared_reshape
// CHECK: %[[SHARED:.*]] = tosa.reshape %arg0{{.*}} -> tensor<2x3xf32>
// CHECK: %[[SUM:.*]] = tosa.add %[[SHARED]], %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK: %[[RESULT:.*]] = tosa.reshape %[[SUM]]{{.*}} -> tensor<1x2x3xf32>
// CHECK: return %[[RESULT]], %[[SHARED]]
func.func @keep_shared_reshape(%arg0: tensor<6xf32>, %arg1: tensor<2x3xf32>) -> (tensor<1x2x3xf32>, tensor<2x3xf32>) {
  %input_shape = tosa.const_shape values(dense<[2, 3]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %shared = tosa.reshape %arg0, %input_shape : (tensor<6xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
  %sum = tosa.add %shared, %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %output_shape = tosa.const_shape values(dense<[1, 2, 3]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %result = tosa.reshape %sum, %output_shape : (tensor<2x3xf32>, !tosa.shape<3>) -> tensor<1x2x3xf32>
  return %result, %shared : tensor<1x2x3xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @propagate_foldable_operands
// CHECK: %[[LHS:.*]] = tosa.reshape %arg0{{.*}} -> tensor<1x2x3x4xf32>
// CHECK: %[[RHS:.*]] = tosa.reshape %arg1{{.*}} -> tensor<1x1x1x4xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[RHS]]{{.*}} : (tensor<1x1x1x4xf16>) -> tensor<1x1x1x4xf32>
// CHECK: %[[ADD:.*]] = tosa.add %[[LHS]], %[[CAST]] : (tensor<1x2x3x4xf32>, tensor<1x1x1x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: return %[[ADD]]
func.func @propagate_foldable_operands(%arg0: tensor<1x6x4xf32>, %arg1: tensor<4xf16>) -> tensor<1x2x3x4xf32> {
  %flat_shape = tosa.const_shape values(dense<[6, 4]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %flat = tosa.reshape %arg0, %flat_shape : (tensor<1x6x4xf32>, !tosa.shape<2>) -> tensor<6x4xf32>
  %bias_shape = tosa.const_shape values(dense<[1, 4]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %bias_2d = tosa.reshape %arg1, %bias_shape : (tensor<4xf16>, !tosa.shape<2>) -> tensor<1x4xf16>
  %bias = tosa.cast %bias_2d : (tensor<1x4xf16>) -> tensor<1x4xf32>
  %add = tosa.add %flat, %bias : (tensor<6x4xf32>, tensor<1x4xf32>) -> tensor<6x4xf32>
  %target_shape = tosa.const_shape values(dense<[1, 2, 3, 4]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %reshape = tosa.reshape %add, %target_shape : (tensor<6x4xf32>, !tosa.shape<4>) -> tensor<1x2x3x4xf32>
  return %reshape : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @keep_shared_producer
// CHECK: %[[CAST:.*]] = tosa.cast %arg0{{.*}} : (tensor<24xf16>) -> tensor<24xf32>
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[CAST]]{{.*}} -> tensor<2x3x4xf32>
// CHECK: return %[[RESHAPE]], %[[CAST]]
func.func @keep_shared_producer(%arg0: tensor<24xf16>) -> (tensor<2x3x4xf32>, tensor<24xf32>) {
  %cast = tosa.cast %arg0 : (tensor<24xf16>) -> tensor<24xf32>
  %shape = tosa.const_shape values(dense<[2, 3, 4]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %cast, %shape : (tensor<24xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  return %reshape, %cast : tensor<2x3x4xf32>, tensor<24xf32>
}

// -----

// CHECK-LABEL: @propagate_non_leading_broadcast
// CHECK-DAG: %[[BIAS:.*]] = tosa.const values(dense<1.000000e+00> : tensor<4x1x1xf32>) : () -> tensor<4x1x1xf32>
// CHECK-DAG: %[[INPUT:.*]] = tosa.reshape %arg0{{.*}} -> tensor<4x2x3xf32>
// CHECK: %[[ADD:.*]] = tosa.add %[[INPUT]], %[[BIAS]] : (tensor<4x2x3xf32>, tensor<4x1x1xf32>) -> tensor<4x2x3xf32>
// CHECK: return %[[ADD]]
func.func @propagate_non_leading_broadcast(%arg0: tensor<4x6xf32>) -> tensor<4x2x3xf32> {
  %bias = "tosa.const"() <{values = dense<1.0> : tensor<4x1xf32>}> : () -> tensor<4x1xf32>
  %add = tosa.add %arg0, %bias : (tensor<4x6xf32>, tensor<4x1xf32>) -> tensor<4x6xf32>
  %shape = tosa.const_shape values(dense<[4, 2, 3]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %reshape = tosa.reshape %add, %shape : (tensor<4x6xf32>, !tosa.shape<3>) -> tensor<4x2x3xf32>
  return %reshape : tensor<4x2x3xf32>
}

// -----

// CHECK-LABEL: @propagate_preserves_properties
// CHECK-DAG: %[[BIAS:.*]] = tosa.const values(dense<1.000000e+00> : tensor<1x1xf32>) : () -> tensor<1x1xf32>
// CHECK-DAG: %[[INPUT:.*]] = tosa.reshape %arg0{{.*}} -> tensor<2x3xf32>
// CHECK: %[[MAX:.*]] = tosa.maximum %[[INPUT]], %[[BIAS]] nan_mode<IGNORE> : (tensor<2x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>
// CHECK: return %[[MAX]]
func.func @propagate_preserves_properties(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  %bias = "tosa.const"() <{values = dense<1.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %maximum = tosa.maximum %arg0, %bias nan_mode<IGNORE> : (tensor<6xf32>, tensor<1xf32>) -> tensor<6xf32>
  %shape = tosa.const_shape values(dense<[2, 3]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %reshape = tosa.reshape %maximum, %shape : (tensor<6xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
  return %reshape : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @fold_single_use_non_splat_constant
// CHECK-DAG: %[[CONST:.*]] = tosa.const values(dense<{{.*}}> : tensor<2x3xf32>) : () -> tensor<2x3xf32>
// CHECK-DAG: %[[INPUT:.*]] = tosa.reshape %arg0{{.*}} -> tensor<2x3xf32>
// CHECK: %[[ADD:.*]] = tosa.add %[[INPUT]], %[[CONST]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK: return %[[ADD]]
func.func @fold_single_use_non_splat_constant(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  %constant = "tosa.const"() <{values = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>}> : () -> tensor<6xf32>
  %add = tosa.add %arg0, %constant : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %shape = tosa.const_shape values(dense<[2, 3]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %reshape = tosa.reshape %add, %shape : (tensor<6xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
  return %reshape : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @keep_shared_non_splat_constant
// CHECK: %[[CONST:.*]] = tosa.const values(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32>) : () -> tensor<6xf32>
// CHECK: %[[ADD:.*]] = tosa.add %arg0, %[[CONST]] : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
// CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[ADD]]{{.*}} -> tensor<2x3xf32>
// CHECK: return %[[RESHAPE]], %[[CONST]]
func.func @keep_shared_non_splat_constant(%arg0: tensor<6xf32>) -> (tensor<2x3xf32>, tensor<6xf32>) {
  %constant = "tosa.const"() <{values = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>}> : () -> tensor<6xf32>
  %add = tosa.add %arg0, %constant : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %shape = tosa.const_shape values(dense<[2, 3]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %reshape = tosa.reshape %add, %shape : (tensor<6xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
  return %reshape, %constant : tensor<2x3xf32>, tensor<6xf32>
}
