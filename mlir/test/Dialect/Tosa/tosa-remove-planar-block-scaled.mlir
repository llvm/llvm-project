// RUN: mlir-opt --split-input-file --verify-diagnostics --tosa-remove-planar-block-scaled %s | FileCheck %s
// RUN: mlir-opt --split-input-file --verify-diagnostics --tosa-remove-planar-block-scaled=convert-function-boundaries %s | FileCheck %s

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_decompose_compose_sequence
// CHECK-SAME: %arg0: tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: return %arg0 : tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @test_decompose_compose_sequence(%input: tensor<2x32x!block_scaled>) -> tensor<2x32x!block_scaled> {
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_shared_decompose
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NOT: tosa.compose_block_scaled
// CHECK: %[[RESULT:.*]] = tosa.concat %arg0, %arg0 axis(0)
// CHECK-NEXT: return %[[RESULT]] : tensor<4x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @test_shared_decompose(%input: tensor<2x32x!block_scaled>) -> tensor<4x32x!block_scaled> {
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %lhs = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  %rhs = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  %result = tosa.concat %lhs, %rhs axis(0) : (tensor<2x32x!block_scaled>, tensor<2x32x!block_scaled>) -> tensor<4x32x!block_scaled>
  return %result : tensor<4x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_multiple_decompose_compose_inputs
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: %[[RESULT:.*]] = tosa.concat %arg0, %arg0 axis(0)
// CHECK-NEXT: return %[[RESULT]] : tensor<4x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @test_multiple_decompose_compose_inputs(%input: tensor<2x32x!block_scaled>) -> tensor<4x32x!block_scaled> {
  %lhs_data, %lhs_scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %rhs_data, %rhs_scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %lhs = tosa.compose_block_scaled %lhs_data, %lhs_scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  %rhs = tosa.compose_block_scaled %rhs_data, %rhs_scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  %result = tosa.concat %lhs, %rhs axis(0) : (tensor<2x32x!block_scaled>, tensor<2x32x!block_scaled>) -> tensor<4x32x!block_scaled>
  return %result : tensor<4x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_transpose_planar_api
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: %[[TRANSPOSE:.*]] = tosa.transpose %arg0 perms([1, 0, 2])
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: return %[[TRANSPOSE]] : tensor<2x2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @test_transpose_planar_api(%input: tensor<2x2x32x!block_scaled>) -> tensor<2x2x32x!block_scaled> {
  %data0, %scale0 = tosa.decompose_block_scaled %input : (tensor<2x2x32x!block_scaled>) -> (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>)
  %packed0 = tosa.compose_block_scaled %data0, %scale0 block_shape<BLOCK_SHAPE_32> : (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>) -> tensor<2x2x32x!block_scaled>
  %transposed = tosa.transpose %packed0 perms([1, 0, 2]) : (tensor<2x2x32x!block_scaled>) -> tensor<2x2x32x!block_scaled>
  %data1, %scale1 = tosa.decompose_block_scaled %transposed : (tensor<2x2x32x!block_scaled>) -> (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>)
  %packed1 = tosa.compose_block_scaled %data1, %scale1 block_shape<BLOCK_SHAPE_32> : (tensor<2x2x32xf8E4M3FN>, tensor<2x2x1xf8E8M0FNU>) -> tensor<2x2x32x!block_scaled>
  return %packed1 : tensor<2x2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>

// CHECK-LABEL: test_decompose_compose_dynamic
// CHECK-SAME: %arg0: tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: return %arg0
func.func @test_decompose_compose_dynamic(%input: tensor<2x?x!block_scaled>) -> tensor<2x?x!block_scaled> {
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x?x!block_scaled>) -> (tensor<2x?xf4E2M1FN>, tensor<2x?xf8E8M0FNU>)
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x?xf4E2M1FN>, tensor<2x?xf8E8M0FNU>) -> tensor<2x?x!block_scaled>
  return %packed : tensor<2x?x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E5M2>

// CHECK-LABEL: test_compose_decompose_unranked
// CHECK-SAME: %arg0: tensor<*x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E5M2>>) -> tensor<*x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E5M2>>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: return %arg0
func.func @test_compose_decompose_unranked(%input: tensor<*x!block_scaled>) -> tensor<*x!block_scaled> {
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<*x!block_scaled>) -> (tensor<*xf8E5M2>, tensor<*xf8E8M0FNU>)
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<*xf8E5M2>, tensor<*xf8E8M0FNU>) -> tensor<*x!block_scaled>
  return %packed : tensor<*x!block_scaled>
}

// -----

// CHECK-LABEL: test_identity
// CHECK: %[[CONST:.*]] = tosa.const values(dense<1.000000e+00> : tensor<1xf16>) : () -> tensor<1xf16>
func.func @test_identity() -> tensor<1xf16> {
  %constant = tosa.const values(dense<1.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
  %result = tosa.cast %constant : (tensor<1xf32>) -> tensor<1xf16>
  return %result : tensor<1xf16>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// CHECK-LABEL: test_matmul_reshape
// CHECK-SAME: %[[LHS:.*]]: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, %[[RHS:.*]]: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: %[[SHAPE:.*]] = tosa.const_shape values(dense<[1, 1, 32, 32]> : tensor<4xindex>) : () -> !tosa.shape<4>
// CHECK-NEXT: %[[ZERO:.*]] = tosa.const values(dense<0.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = tosa.matmul_t %[[LHS]], %[[RHS]], %[[ZERO]], %[[ZERO]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
// CHECK-NEXT: %[[PACKED:.*]] = tosa.cast %[[MATMUL]] : (tensor<1x32x32xf32>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NEXT: %[[RESHAPED:.*]] = tosa.reshape %[[PACKED]], %[[SHAPE]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, !tosa.shape<4>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// CHECK-NEXT: %[[RESULT:.*]] = tosa.cast %[[RESHAPED]] : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<1x1x32x32xf32>
// CHECK-NOT: tosa.compose_block_scaled
// CHECK-NOT: tosa.decompose_block_scaled
// CHECK-NEXT: }
func.func @test_matmul_reshape(%arg0: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, %arg1: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32> {
  %data, %scale = tosa.decompose_block_scaled %arg1 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
  %data_0, %scale_1 = tosa.decompose_block_scaled %arg0 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
  %0 = tosa.const  values(dense<0.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
  %1 = tosa.compose_block_scaled %data_0, %scale_1 block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %2 = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %3 = tosa.matmul_t %1, %2, %0, %0 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
  %4 = tosa.cast %3  : (tensor<1x32x32xf32>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %data_2, %scale_3 = tosa.decompose_block_scaled %4 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
  %5 = tosa.const_shape  values(dense<[1, 1, 32, 32]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %6 = tosa.compose_block_scaled %data_2, %scale_3 block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %7 = tosa.reshape %6, %5 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, !tosa.shape<4>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %data_4, %scale_5 = tosa.decompose_block_scaled %7 : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x1xf8E8M0FNU>)
  %8 = tosa.compose_block_scaled %data_4, %scale_5 block_shape<BLOCK_SHAPE_32> : (tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x1xf8E8M0FNU>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  %9 = tosa.cast %8  : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
  return %9 : tensor<1x1x32x32xf32>
}
