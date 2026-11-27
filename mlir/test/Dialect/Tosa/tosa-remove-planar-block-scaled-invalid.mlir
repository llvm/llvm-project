// RUN: mlir-opt --split-input-file --verify-diagnostics --tosa-remove-planar-block-scaled %s

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_unpaired_compose(%data: tensor<2x32xf8E4M3FN>, %scale: tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled> {
  // expected-error@+1 {{failed to legalize operation 'tosa.compose_block_scaled'}}
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_unpaired_decompose(%input: tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) {
  // expected-error@+1 {{failed to legalize operation 'tosa.decompose_block_scaled'}}
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  return %data, %scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_already_block_scaled(%lhs: tensor<2x32x!block_scaled>, %rhs: tensor<2x32x!block_scaled>) -> tensor<2x32x!block_scaled> {
  // expected-error@+1 {{failed to legalize operation 'tosa.decompose_block_scaled'}}
  %lhs_data, %lhs_scale = tosa.decompose_block_scaled %lhs : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %rhs_data, %rhs_scale = tosa.decompose_block_scaled %rhs : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %packed = tosa.compose_block_scaled %lhs_data, %rhs_scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_live_planar_user(%input: tensor<2x32x!block_scaled>) -> (tensor<2x32x!block_scaled>, tensor<2x32xf8E4M3FN>) {
  // expected-error@+1 {{failed to legalize operation 'tosa.decompose_block_scaled'}}
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed, %data : tensor<2x32x!block_scaled>, tensor<2x32xf8E4M3FN>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_duplicate_results(%input: tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>, tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) {
  // expected-error@+1 {{failed to legalize operation 'tosa.decompose_block_scaled'}}
  %data, %scale = tosa.decompose_block_scaled %input : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  return %data, %scale, %data, %scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>, tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>
}
