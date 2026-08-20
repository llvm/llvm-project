// RUN: mlir-opt --split-input-file --verify-diagnostics --tosa-remove-planar-block-scaled=convert-function-boundaries %s

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_compose_scale_from_operation(%data: tensor<2x32xf8E4M3FN>, %scale: tensor<1x2xf8E8M0FNU>) -> tensor<2x32x!block_scaled> {
  %transposed = tosa.transpose %scale perms([1, 0]) : (tensor<1x2xf8E8M0FNU>) -> tensor<2x1xf8E8M0FNU>
  // expected-error@+1 {{requires data and scale operands to be function entry arguments}}
  %packed = tosa.compose_block_scaled %data, %transposed block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_compose_non_entry_arguments(%condition: i1, %lhs_data: tensor<2x32xf8E4M3FN>, %lhs_scale: tensor<2x1xf8E8M0FNU>, %rhs_data: tensor<2x32xf8E4M3FN>, %rhs_scale: tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled> {
  cf.cond_br %condition, ^bb1(%lhs_data, %lhs_scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>), ^bb1(%rhs_data, %rhs_scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
^bb1(%data: tensor<2x32xf8E4M3FN>, %scale: tensor<2x1xf8E8M0FNU>):
  // expected-error@+1 {{requires data and scale operands to be function entry arguments}}
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed : tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// expected-error@+1 {{requires planar arguments to be used by a single compose}}
func.func @test_compose_data_has_another_use(%data: tensor<2x32xf8E4M3FN>, %scale: tensor<2x1xf8E8M0FNU>) -> (tensor<2x32x!block_scaled>, tensor<2x32xf8E4M3FN>) {
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed, %data : tensor<2x32x!block_scaled>, tensor<2x32xf8E4M3FN>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// expected-error@+1 {{requires planar arguments to be used by a single compose}}
func.func @test_compose_scale_has_another_use(%data: tensor<2x32xf8E4M3FN>, %scale: tensor<2x1xf8E8M0FNU>) -> (tensor<2x32x!block_scaled>, tensor<2x1xf8E8M0FNU>) {
  %packed = tosa.compose_block_scaled %data, %scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %packed, %scale : tensor<2x32x!block_scaled>, tensor<2x1xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

// expected-error@+1 {{requires planar arguments to be used by a single compose}}
func.func @test_argument_shared_by_composes(%data: tensor<2x32xf8E4M3FN>, %lhs_scale: tensor<2x1xf8E8M0FNU>, %rhs_scale: tensor<2x1xf8E8M0FNU>) -> (tensor<2x32x!block_scaled>, tensor<2x32x!block_scaled>) {
  %lhs = tosa.compose_block_scaled %data, %lhs_scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  %rhs = tosa.compose_block_scaled %data, %rhs_scale block_shape<BLOCK_SHAPE_32> : (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) -> tensor<2x32x!block_scaled>
  return %lhs, %rhs : tensor<2x32x!block_scaled>, tensor<2x32x!block_scaled>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_decompose_only_data_returned(%packed: tensor<2x32x!block_scaled>) -> tensor<2x32xf8E4M3FN> {
  // expected-error@+1 {{requires both data and scale results to be returned}}
  %data, %scale = tosa.decompose_block_scaled %packed : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  return %data : tensor<2x32xf8E4M3FN>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_decompose_only_scale_returned(%packed: tensor<2x32x!block_scaled>) -> tensor<2x1xf8E8M0FNU> {
  // expected-error@+1 {{requires both data and scale results to be returned}}
  %data, %scale = tosa.decompose_block_scaled %packed : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  return %scale : tensor<2x1xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_decompose_data_used_by_operation(%packed: tensor<2x32x!block_scaled>) -> (tensor<32x2xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) {
  // expected-error@+1 {{requires data and scale results to be used only by returns of the enclosing function}}
  %data, %scale = tosa.decompose_block_scaled %packed : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %transposed = tosa.transpose %data perms([1, 0]) : (tensor<2x32xf8E4M3FN>) -> tensor<32x2xf8E4M3FN>
  return %transposed, %scale : tensor<32x2xf8E4M3FN>, tensor<2x1xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func @test_decompose_scale_used_by_operation(%packed: tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<1x2xf8E8M0FNU>) {
  // expected-error@+1 {{requires data and scale results to be used only by returns of the enclosing function}}
  %data, %scale = tosa.decompose_block_scaled %packed : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  %transposed = tosa.transpose %scale perms([1, 0]) : (tensor<2x1xf8E8M0FNU>) -> tensor<1x2xf8E8M0FNU>
  return %data, %transposed : tensor<2x32xf8E4M3FN>, tensor<1x2xf8E8M0FNU>
}

// -----

!block_scaled = !tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>

func.func private @side_effect()

// expected-error@+1 {{requires a single function return when rewriting planar results}}
func.func @test_decompose_multiple_function_returns(%condition: i1, %packed: tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>) {
  %data, %scale = tosa.decompose_block_scaled %packed : (tensor<2x32x!block_scaled>) -> (tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>)
  cf.cond_br %condition, ^bb1, ^bb2
^bb1:
  func.call @side_effect() : () -> ()
  return %data, %scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>
^bb2:
  return %data, %scale : tensor<2x32xf8E4M3FN>, tensor<2x1xf8E8M0FNU>
}
