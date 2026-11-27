// RUN: mlir-opt --split-input-file --tosa-remove-planar-block-scaled=convert-function-boundaries %s | FileCheck %s

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_convert_input_simple
// CHECK-SAME: %arg0: tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NEXT: return %arg0 : tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @test_convert_input_simple(%data: tensor<2x32xf8E4M3FN>, %scale: tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled> {
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_shuffled_input_arguments
// CHECK-SAME: %arg0: i32, %arg1: tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NEXT: return %arg1
func.func @test_shuffled_input_arguments(%scale: tensor<2x1xf8E8M0FNU>, %count: i32, %data: tensor<2x32xf8E4M3FN>) -> tensor<2x32x!block_scaled> {
  %result = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %result : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_convert_input_multiple
// CHECK-SAME: %[[FIRST:.*]]: tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, %[[SECOND:.*]]: tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>)
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NEXT: return %[[FIRST]], %[[SECOND]] : tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NEXT: }
func.func @test_convert_input_multiple(%data_1: tensor<2x32xf8E4M3FN>, %scale_1: tensor<2x1xf8E8M0FNU>, %data_2: tensor<2x32xf8E4M3FN>, %scale_2: tensor<2x1xf8E8M0FNU>) -> (tensor<2x32x!block_scaled>, tensor<2x32x!block_scaled>) {
  %packed_1 = tosa.compose_block_scaled %data_1, %scale_1 block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  %packed_2 = tosa.compose_block_scaled %data_2, %scale_2 block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed_1, %packed_2 : tensor<2x32x!block_scaled>, tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_convert_result_simple
// CHECK-SAME: %arg0: tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: return %arg0 : tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @test_convert_result_simple(%input: tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) {
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  return %data, %scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_transpose_with_planar_api
// CHECK-SAME: %arg0: tensor<2x2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<2x2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NEXT: %[[TRANSPOSE:.*]] = tosa.transpose %arg0 perms([1, 0, 2])
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: return %[[TRANSPOSE]]
func.func @test_transpose_with_planar_api(%data: tensor<2x2x32xf8E4M3FN>, %scale: tensor<2x2x1xf8E8M0FNU>) -> (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>) {
  %input = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>) -> tensor<2x2x32x!block_scaled>
  %result = tosa.transpose %input perms([1, 0, 2]) : (tensor<2x2x32x!block_scaled>) -> tensor<2x2x32x!block_scaled>
  %result_data, %result_scale = tosa.decompose_block_scaled %result : (tensor<2x2x32x!block_scaled>) -> (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>)
  return %result_data, %result_scale : tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_matmul_reshape
// CHECK-SAME: %[[LHS:.*]]: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, %[[RHS:.*]]: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
// CHECK-NEXT: %[[SHAPE:.*]] = tosa.const_shape values(dense<[1, 1, 32, 32]> : tensor<4xindex>) : () -> !tosa.shape<4>
// CHECK-NEXT: %[[ZERO:.*]] = tosa.const values(dense<0.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = tosa.matmul_t %[[LHS]], %[[RHS]], %[[ZERO]], %[[ZERO]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
// CHECK-NEXT: %[[PACKED:.*]] = tosa.cast %[[MATMUL]] : (tensor<1x32x32xf32>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NEXT: %[[RESHAPED:.*]] = tosa.reshape %[[PACKED]], %[[SHAPE]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, !tosa.shape<4>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NEXT: %[[RESULT:.*]] = tosa.cast %[[RESHAPED]] : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<1x1x32x32xf32>
func.func @test_matmul_reshape(%arg0: tensor<1x32x32xf8E4M3FN>, %arg1: tensor<1x32x1xf8E8M0FNU>, %arg2: tensor<1x32x32xf8E4M3FN>, %arg3: tensor<1x32x1xf8E8M0FNU>) -> tensor<1x1x32x32xf32> {
  %0 = tosa.const  values(dense<0.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
  %1 = tosa.compose_block_scaled %arg0, %arg1 block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %2 = tosa.compose_block_scaled %arg2, %arg3 block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %3 = tosa.matmul_t %1, %2, %0, %0 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
  %4 = tosa.cast %3  : (tensor<1x32x32xf32>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %data, %scale = tosa.decompose_block_scaled %4 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
  %5 = tosa.const_shape  values(dense<[1, 1, 32, 32]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %6 = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %7 = tosa.reshape %6, %5 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, !tosa.shape<4>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %data_0, %scale_1 = tosa.decompose_block_scaled %7 : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x1xf8E8M0FNU>)
  %8 = tosa.compose_block_scaled %data_0, %scale_1 block_shape<BLOCK_SHAPE_32> : (tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x1xf8E8M0FNU>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %9 = tosa.cast %8  : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
  return %9 : tensor<1x1x32x32xf32>
}
