// RUN: mlir-opt --split-input-file --tosa-materialize-planar-block-scaled %s | FileCheck %s --check-prefixes=COMMON,DEFAULT
// RUN: mlir-opt --split-input-file '--tosa-materialize-planar-block-scaled=convert-function-boundaries' %s | FileCheck %s --check-prefixes=COMMON,BOUNDARY

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// COMMON-LABEL: test_matmul
// DEFAULT-SAME: %arg0: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, %arg1: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x32x32xf32>
// BOUNDARY-SAME: %arg0: tensor<1x32x32xf8E4M3FN>, %arg1: tensor<1x32x1xf8E8M0FNU>, %arg2: tensor<1x32x32xf8E4M3FN>, %arg3: tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32xf32>
// DEFAULT: %[[DATA_B:.*]], %[[SCALE_B:.*]] = tosa.decompose_block_scaled %arg1 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
// DEFAULT: %[[DATA_A:.*]], %[[SCALE_A:.*]] = tosa.decompose_block_scaled %arg0 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
// COMMON: %[[ZERO:.*]] = tosa.const values(dense<0.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
// DEFAULT: %[[A_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %[[DATA_A]], %[[SCALE_A]]
// DEFAULT: %[[B_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %[[DATA_B]], %[[SCALE_B]]
// BOUNDARY: %[[A_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %arg0, %arg1
// BOUNDARY: %[[B_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %arg2, %arg3
// COMMON: %[[MATMUL_RESULT:.*]] = tosa.matmul_t %[[A_BLOCK_SCALED]], %[[B_BLOCK_SCALED]], %[[ZERO]], %[[ZERO]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
// COMMON: return %[[MATMUL_RESULT]] : tensor<1x32x32xf32>
func.func @test_matmul(%arg0: tensor<1x32x32x!block_scaled>, %arg1: tensor<1x32x32x!block_scaled>) -> tensor<1x32x32xf32> {
  %c0 = tosa.const values(dense<0.0> : tensor<1xf32>) : () -> tensor<1xf32>
  %0 = tosa.matmul_t %arg0, %arg1, %c0, %c0 :
      (tensor<1x32x32x!block_scaled>, tensor<1x32x32x!block_scaled>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
  return %0 : tensor<1x32x32xf32>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// COMMON-LABEL: test_matmul_reshape
// DEFAULT-SAME: %arg0: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, %arg1: tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
// BOUNDARY-SAME: %arg0: tensor<1x32x32xf8E4M3FN>, %arg1: tensor<1x32x1xf8E8M0FNU>, %arg2: tensor<1x32x32xf8E4M3FN>, %arg3: tensor<1x32x1xf8E8M0FNU>) -> tensor<1x1x32x32xf32>
// DEFAULT: %[[DATA_B:.*]], %[[SCALE_B:.*]] = tosa.decompose_block_scaled %arg1 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
// DEFAULT: %[[DATA_A:.*]], %[[SCALE_A:.*]] = tosa.decompose_block_scaled %arg0 : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
// COMMON: %[[ZERO:.*]] = tosa.const values(dense<0.000000e+00> : tensor<1xf32>) : () -> tensor<1xf32>
// DEFAULT: %[[A_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %[[DATA_A]], %[[SCALE_A]]
// DEFAULT: %[[B_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %[[DATA_B]], %[[SCALE_B]]
// BOUNDARY: %[[A_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %arg0, %arg1
// BOUNDARY: %[[B_BLOCK_SCALED:.*]] = tosa.compose_block_scaled %arg2, %arg3
// COMMON: %[[MATMUL:.*]] = tosa.matmul_t %[[A_BLOCK_SCALED]], %[[B_BLOCK_SCALED]], %[[ZERO]], %[[ZERO]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
// COMMON: %[[CAST:.*]] = tosa.cast %[[MATMUL]] : (tensor<1x32x32xf32>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// COMMON: %[[PLANAR_CAST_DATA:.*]], %[[PLANAR_CAST_SCALE:.*]] = tosa.decompose_block_scaled %[[CAST]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>)
// COMMON: %[[NEW_SHAPE:.*]] = tosa.const_shape values(dense<[1, 1, 32, 32]> : tensor<4xindex>) : () -> !tosa.shape<4>
// COMMON: %[[RESHAPE_IN:.*]] = tosa.compose_block_scaled %[[PLANAR_CAST_DATA]], %[[PLANAR_CAST_SCALE]] block_shape<BLOCK_SHAPE_32> : (tensor<1x32x32xf8E4M3FN>, tensor<1x32x1xf8E8M0FNU>) -> tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// COMMON: %[[RESHAPE:.*]] = tosa.reshape %[[RESHAPE_IN]], %[[NEW_SHAPE]] : (tensor<1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>, !tosa.shape<4>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// COMMON: %[[PLANAR_RESHAPE_DATA:.*]], %[[PLANAR_RESHAPE_SCALE:.*]] = tosa.decompose_block_scaled %[[RESHAPE]] : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> (tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x1xf8E8M0FNU>)
// COMMON: %[[CAST2_IN:.*]] = tosa.compose_block_scaled %[[PLANAR_RESHAPE_DATA]], %[[PLANAR_RESHAPE_SCALE]] block_shape<BLOCK_SHAPE_32> : (tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x1xf8E8M0FNU>) -> tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// COMMON: %[[CAST2:.*]] = tosa.cast %[[CAST2_IN]] : (tensor<1x1x32x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x1x32x32xf32>
// COMMON: return %[[CAST2]] : tensor<1x1x32x32xf32>
func.func @test_matmul_reshape(%a: tensor<1x32x32x!block_scaled>, %b: tensor<1x32x32x!block_scaled>) -> tensor<1x1x32x32xf32> {
  %c0 = tosa.const values(dense<0.0> : tensor<1xf32>) : () -> tensor<1xf32>
  %mm = tosa.matmul_t %a, %b, %c0, %c0 :
      (tensor<1x32x32x!block_scaled>, tensor<1x32x32x!block_scaled>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32xf32>
  %cast = tosa.cast %mm :
      (tensor<1x32x32xf32>) -> tensor<1x32x32x!block_scaled>
  %shape = tosa.const_shape values(dense<[1, 1, 32, 32]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %reshape = tosa.reshape %cast, %shape :
      (tensor<1x32x32x!block_scaled>, !tosa.shape<4>) -> tensor<1x1x32x32x!block_scaled>
  %result = tosa.cast %reshape :
      (tensor<1x1x32x32x!block_scaled>) -> tensor<1x1x32x32xf32>
  return %result : tensor<1x1x32x32xf32>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// COMMON-LABEL: test_cond_if
// DEFAULT-SAME: %arg0: tensor<i1>, %arg1: tensor<1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>) -> tensor<1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// BOUNDARY-SAME: %arg0: tensor<i1>, %arg1: tensor<1x32xf8E4M3FN>, %arg2: tensor<1x1xf8E8M0FNU>) -> (tensor<1x32xf8E4M3FN>, tensor<1x1xf8E8M0FNU>)
// DEFAULT: %[[INPUT_DATA:.*]], %[[INPUT_SCALE:.*]] = tosa.decompose_block_scaled %arg1
// DEFAULT: %[[COND_INPUT:.*]] = tosa.compose_block_scaled %[[INPUT_DATA]], %[[INPUT_SCALE]]
// BOUNDARY: %[[COND_INPUT:.*]] = tosa.compose_block_scaled %arg1, %arg2
// COMMON: %[[COND_RESULT:.*]] = tosa.cond_if %arg0
// COMMON-SAME: = %[[COND_INPUT]]
// COMMON: ^bb0(%[[THEN_ARG:.*]]: tensor<1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>):
// COMMON: %[[THEN_DATA:.*]], %[[THEN_SCALE:.*]] = tosa.decompose_block_scaled %[[THEN_ARG]]
// COMMON: %[[THEN_YIELD:.*]] = tosa.compose_block_scaled %[[THEN_DATA]], %[[THEN_SCALE]]
// COMMON: tosa.yield %[[THEN_YIELD]]
// COMMON: } else {
// COMMON: ^bb0(%[[ELSE_ARG:.*]]: tensor<1x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>):
// COMMON: %[[ELSE_DATA:.*]], %[[ELSE_SCALE:.*]] = tosa.decompose_block_scaled %[[ELSE_ARG]]
// COMMON: %[[ELSE_YIELD:.*]] = tosa.compose_block_scaled %[[ELSE_DATA]], %[[ELSE_SCALE]]
// COMMON: tosa.yield %[[ELSE_YIELD]]
// COMMON: %[[RESULT_DATA:.*]], %[[RESULT_SCALE:.*]] = tosa.decompose_block_scaled %[[COND_RESULT]]
// DEFAULT: %[[RESULT_PACKED:.*]] = tosa.compose_block_scaled %[[RESULT_DATA]], %[[RESULT_SCALE]]
// DEFAULT: return %[[RESULT_PACKED]]
// BOUNDARY: return %[[RESULT_DATA]], %[[RESULT_SCALE]]
func.func @test_cond_if(%condition: tensor<i1>, %value: tensor<1x32x!block_scaled>) -> tensor<1x32x!block_scaled> {
  %result = tosa.cond_if %condition (%branch_arg = %value) : tensor<i1> (tensor<1x32x!block_scaled>) -> tensor<1x32x!block_scaled> {
  ^bb0(%branch_arg: tensor<1x32x!block_scaled>):
    tosa.yield %branch_arg : tensor<1x32x!block_scaled>
  } else {
  ^bb0(%branch_arg: tensor<1x32x!block_scaled>):
    tosa.yield %branch_arg : tensor<1x32x!block_scaled>
  }
  return %result : tensor<1x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>

// COMMON-LABEL: test_concat
// DEFAULT-SAME: %arg0: tensor<13x21x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>, %arg1: tensor<13x21x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<26x21x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-SAME: %arg0: tensor<13x21x32xf4E2M1FN>, %arg1: tensor<13x21x1xf8E8M0FNU>, %arg2: tensor<13x21x32xf4E2M1FN>, %arg3: tensor<13x21x1xf8E8M0FNU>) -> (tensor<26x21x32xf4E2M1FN>, tensor<26x21x1xf8E8M0FNU>)
// DEFAULT: %[[RHS_DATA:.*]], %[[RHS_SCALE:.*]] = tosa.decompose_block_scaled %arg1
// DEFAULT: %[[LHS_DATA:.*]], %[[LHS_SCALE:.*]] = tosa.decompose_block_scaled %arg0
// DEFAULT: %[[CONCAT_LHS:.*]] = tosa.compose_block_scaled %[[LHS_DATA]], %[[LHS_SCALE]]
// DEFAULT: %[[CONCAT_RHS:.*]] = tosa.compose_block_scaled %[[RHS_DATA]], %[[RHS_SCALE]]
// BOUNDARY: %[[CONCAT_LHS:.*]] = tosa.compose_block_scaled %arg0, %arg1
// BOUNDARY: %[[CONCAT_RHS:.*]] = tosa.compose_block_scaled %arg2, %arg3
// COMMON: %[[CONCAT_RESULT:.*]] = tosa.concat %[[CONCAT_LHS]], %[[CONCAT_RHS]] axis(0)
// COMMON: %[[RESULT_DATA:.*]], %[[RESULT_SCALE:.*]] = tosa.decompose_block_scaled %[[CONCAT_RESULT]]
// DEFAULT: %[[RESULT_PACKED:.*]] = tosa.compose_block_scaled %[[RESULT_DATA]], %[[RESULT_SCALE]]
// DEFAULT: return %[[RESULT_PACKED]]
// BOUNDARY: return %[[RESULT_DATA]], %[[RESULT_SCALE]]
func.func @test_concat(%lhs: tensor<13x21x32x!block_scaled>, %rhs: tensor<13x21x32x!block_scaled>) -> tensor<26x21x32x!block_scaled> {
  %result = tosa.concat %lhs, %rhs axis(0) :
      (tensor<13x21x32x!block_scaled>, tensor<13x21x32x!block_scaled>) -> tensor<26x21x32x!block_scaled>
  return %result : tensor<26x21x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>

// COMMON-LABEL: test_transpose
// DEFAULT-SAME: %arg0: tensor<13x21x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<21x13x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-SAME: %arg0: tensor<13x21x32xf4E2M1FN>, %arg1: tensor<13x21x1xf8E8M0FNU>) -> (tensor<21x13x32xf4E2M1FN>, tensor<21x13x1xf8E8M0FNU>)
// DEFAULT: %[[INPUT_DATA:.*]], %[[INPUT_SCALE:.*]] = tosa.decompose_block_scaled %arg0
// DEFAULT: %[[TRANSPOSE_INPUT:.*]] = tosa.compose_block_scaled %[[INPUT_DATA]], %[[INPUT_SCALE]]
// BOUNDARY: %[[TRANSPOSE_INPUT:.*]] = tosa.compose_block_scaled %arg0, %arg1
// COMMON: %[[TRANSPOSE_RESULT:.*]] = tosa.transpose %[[TRANSPOSE_INPUT]] perms([1, 0, 2])
// COMMON: %[[RESULT_DATA:.*]], %[[RESULT_SCALE:.*]] = tosa.decompose_block_scaled %[[TRANSPOSE_RESULT]]
// DEFAULT: %[[RESULT_PACKED:.*]] = tosa.compose_block_scaled %[[RESULT_DATA]], %[[RESULT_SCALE]]
// DEFAULT: return %[[RESULT_PACKED]]
// BOUNDARY: return %[[RESULT_DATA]], %[[RESULT_SCALE]]
func.func @test_transpose(%input: tensor<13x21x32x!block_scaled>) -> tensor<21x13x32x!block_scaled> {
  %result = tosa.transpose %input perms([1, 0, 2]) :
      (tensor<13x21x32x!block_scaled>) -> tensor<21x13x32x!block_scaled>
  return %result : tensor<21x13x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>

// COMMON-LABEL: boundary_callee
// DEFAULT-SAME: %arg0: tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-SAME: %[[CALLEE_DATA:.*]]: tensor<2x?xf4E2M1FN>, %[[CALLEE_SCALE:.*]]: tensor<2x?xf8E8M0FNU>) -> (tensor<2x?xf4E2M1FN>, tensor<2x?xf8E8M0FNU>)
// DEFAULT: return %arg0 : tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY: return %[[CALLEE_DATA]], %[[CALLEE_SCALE]]
func.func private @boundary_callee(%value: tensor<2x?x!block_scaled>) -> tensor<2x?x!block_scaled> {
  return %value : tensor<2x?x!block_scaled>
}

// COMMON-LABEL: test_function_boundaries
// DEFAULT-SAME: %arg0: tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-SAME: %[[CALLER_DATA:.*]]: tensor<2x?xf4E2M1FN>, %[[CALLER_SCALE:.*]]: tensor<2x?xf8E8M0FNU>) -> (tensor<2x?xf4E2M1FN>, tensor<2x?xf8E8M0FNU>)
// DEFAULT: %[[DEFAULT_CALL:.*]] = call @boundary_callee(%arg0) : (tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// DEFAULT: return %[[DEFAULT_CALL]] : tensor<2x?x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY: %[[CALL:.*]]:2 = call @boundary_callee(%[[CALLER_DATA]], %[[CALLER_SCALE]]) : (tensor<2x?xf4E2M1FN>, tensor<2x?xf8E8M0FNU>) -> (tensor<2x?xf4E2M1FN>, tensor<2x?xf8E8M0FNU>)
// BOUNDARY: return %[[CALL]]#0, %[[CALL]]#1
func.func @test_function_boundaries(%value: tensor<2x?x!block_scaled>) -> tensor<2x?x!block_scaled> {
  %result = call @boundary_callee(%value) :
      (tensor<2x?x!block_scaled>) -> tensor<2x?x!block_scaled>
  return %result : tensor<2x?x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>

// COMMON-LABEL: test_identity_multiple_scale_channel_blocks
// DEFAULT-SAME: %arg0: tensor<?x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> tensor<?x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-SAME: %arg0: tensor<?x64xf4E2M1FN>, %arg1: tensor<?x2xf8E8M0FNU>) -> (tensor<?x64xf4E2M1FN>, tensor<?x2xf8E8M0FNU>)
// DEFAULT: %[[INPUT_DATA:.*]], %[[INPUT_SCALE:.*]] = tosa.decompose_block_scaled %arg0 : (tensor<?x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> (tensor<?x64xf4E2M1FN>, tensor<?x2xf8E8M0FNU>)
// DEFAULT: %[[IDENTITY_INPUT:.*]] = tosa.compose_block_scaled %[[INPUT_DATA]], %[[INPUT_SCALE]]
// BOUNDARY: %[[IDENTITY_INPUT:.*]] = tosa.compose_block_scaled %arg0, %arg1
// COMMON: %[[IDENTITY_RESULT:.*]] = tosa.identity %[[IDENTITY_INPUT]]
// COMMON: %[[RESULT_DATA:.*]], %[[RESULT_SCALE:.*]] = tosa.decompose_block_scaled %[[IDENTITY_RESULT]] : (tensor<?x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>) -> (tensor<?x64xf4E2M1FN>, tensor<?x2xf8E8M0FNU>)
// DEFAULT: %[[RESULT_PACKED:.*]] = tosa.compose_block_scaled %[[RESULT_DATA]], %[[RESULT_SCALE]]
// DEFAULT: return %[[RESULT_PACKED]]
// BOUNDARY: return %[[RESULT_DATA]], %[[RESULT_SCALE]]
func.func @test_identity_multiple_scale_channel_blocks(%input: tensor<?x64x!block_scaled>) -> tensor<?x64x!block_scaled> {
  %result = tosa.identity %input : (tensor<?x64x!block_scaled>) -> tensor<?x64x!block_scaled>
  return %result : tensor<?x64x!block_scaled>
}

// -----

// COMMON-LABEL: @test_const_multiple_blocks
// DEFAULT-SAME: () -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// BOUNDARY-SAME: () -> (tensor<2x64xf8E4M3FN>, tensor<2x2xf8E8M0FNU>)
// COMMON-NEXT: %[[DATA:.*]] = tosa.const values(dense<1.000000e+00> : tensor<2x64xf8E4M3FN>) : () -> tensor<2x64xf8E4M3FN>
// COMMON-NEXT: %[[SCALE:.*]] = tosa.const values(dense<{{\[\[}}1.000000e+00, 2.000000e+00], [4.000000e+00, 8.000000e+00]]> : tensor<2x2xf8E8M0FNU>) : () -> tensor<2x2xf8E8M0FNU>
// DEFAULT-NEXT: %[[PACKED:.*]] = tosa.compose_block_scaled %[[DATA]], %[[SCALE]] block_shape<BLOCK_SHAPE_32> : (tensor<2x64xf8E4M3FN>, tensor<2x2xf8E8M0FNU>) -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// DEFAULT-NEXT: return %[[PACKED]] : tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
// BOUNDARY-NEXT: return %[[DATA]], %[[SCALE]] : tensor<2x64xf8E4M3FN>, tensor<2x2xf8E8M0FNU>
func.func @test_const_multiple_blocks() -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>> {
  %packed = tosa.const values(dense<tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN, {1.000000e+00, 2.000000e+00, 4.000000e+00, 8.000000e+00}>> : 1.0 : f8E4M3FN>) : () -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  return %packed : tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
}

// -----

// COMMON-LABEL: @test_const_splat_scale
// DEFAULT-SAME: () -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN>>
// BOUNDARY-SAME: () -> (tensor<2x64xf6E2M3FN>, tensor<2x2xf8E8M0FNU>)
// COMMON-NEXT: %[[DATA:.*]] = tosa.const values(dense<-1.000000e+00> : tensor<2x64xf6E2M3FN>) : () -> tensor<2x64xf6E2M3FN>
// COMMON-NEXT: %[[SCALE:.*]] = tosa.const values(dense<2.000000e+00> : tensor<2x2xf8E8M0FNU>) : () -> tensor<2x2xf8E8M0FNU>
// DEFAULT-NEXT: %[[PACKED:.*]] = tosa.compose_block_scaled %[[DATA]], %[[SCALE]] block_shape<BLOCK_SHAPE_32> : (tensor<2x64xf6E2M3FN>, tensor<2x2xf8E8M0FNU>) -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN>>
// DEFAULT-NEXT: return %[[PACKED]] : tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN>>
// BOUNDARY-NEXT: return %[[DATA]], %[[SCALE]] : tensor<2x64xf6E2M3FN>, tensor<2x2xf8E8M0FNU>
func.func @test_const_splat_scale() -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN>> {
  %packed = tosa.const values(dense<tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN, {2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00}>> : -1.0 : f6E2M3FN>) : () -> tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN>>
  return %packed : tensor<2x64x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f6E2M3FN>>
}

// -----

// COMMON-LABEL: @test_const_subbyte_data
// DEFAULT-SAME: () -> tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-SAME: () -> (tensor<32xf4E2M1FN>, tensor<1xf8E8M0FNU>)
// COMMON-NEXT: %[[DATA:.*]] = tosa.const values(dense<[0.000000e+00, -0.000000e+00, 5.000000e-01, -5.000000e-01, 1.000000e+00, -1.000000e+00, 2.000000e+00, -2.000000e+00, 0.000000e+00, -0.000000e+00, 5.000000e-01, -5.000000e-01, 1.000000e+00, -1.000000e+00, 2.000000e+00, -2.000000e+00, 0.000000e+00, -0.000000e+00, 5.000000e-01, -5.000000e-01, 1.000000e+00, -1.000000e+00, 2.000000e+00, -2.000000e+00, 0.000000e+00, -0.000000e+00, 5.000000e-01, -5.000000e-01, 1.000000e+00, -1.000000e+00, 2.000000e+00, -2.000000e+00]> : tensor<32xf4E2M1FN>) : () -> tensor<32xf4E2M1FN>
// COMMON-NEXT: %[[SCALE:.*]] = tosa.const values(dense<1.000000e+00> : tensor<1xf8E8M0FNU>) : () -> tensor<1xf8E8M0FNU>
// DEFAULT-NEXT: %[[PACKED:.*]] = tosa.compose_block_scaled %[[DATA]], %[[SCALE]] block_shape<BLOCK_SHAPE_32> : (tensor<32xf4E2M1FN>, tensor<1xf8E8M0FNU>) -> tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// DEFAULT-NEXT: return %[[PACKED]] : tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
// BOUNDARY-NEXT: return %[[DATA]], %[[SCALE]] : tensor<32xf4E2M1FN>, tensor<1xf8E8M0FNU>
func.func @test_const_subbyte_data() -> tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>> {
  %packed = tosa.const values(dense<tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN, {1.000000e+00}>> : [0.0 : f4E2M1FN, -0.0 : f4E2M1FN, 0.5 : f4E2M1FN, -0.5 : f4E2M1FN, 1.0 : f4E2M1FN, -1.0 : f4E2M1FN, 2.0 : f4E2M1FN, -2.0 : f4E2M1FN, 0.0 : f4E2M1FN, -0.0 : f4E2M1FN, 0.5 : f4E2M1FN, -0.5 : f4E2M1FN, 1.0 : f4E2M1FN, -1.0 : f4E2M1FN, 2.0 : f4E2M1FN, -2.0 : f4E2M1FN, 0.0 : f4E2M1FN, -0.0 : f4E2M1FN, 0.5 : f4E2M1FN, -0.5 : f4E2M1FN, 1.0 : f4E2M1FN, -1.0 : f4E2M1FN, 2.0 : f4E2M1FN, -2.0 : f4E2M1FN, 0.0 : f4E2M1FN, -0.0 : f4E2M1FN, 0.5 : f4E2M1FN, -0.5 : f4E2M1FN, 1.0 : f4E2M1FN, -1.0 : f4E2M1FN, 2.0 : f4E2M1FN, -2.0 : f4E2M1FN]>) : () -> tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
  return %packed : tensor<32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f4E2M1FN>>
}
