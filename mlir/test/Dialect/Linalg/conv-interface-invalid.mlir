// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @test_conv_op_not_linalg_op(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>,
    %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected a LinalgOp}}
  %0 = "test.conv_op_not_linalg_op"(%arg0, %arg1, %arg2)
      : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Check for number of operands being >= 2.
#map = affine_map<(d0) -> (d0)>
func.func @test_conv_op_wrong_num_operands(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected op with 2 inputs and 1 output}}
  %0 = test.linalg_conv_op {
      indexing_maps = [#map, #map],
      iterator_types = [#test.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32):
         linalg.yield  %arg3 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_conv_op_wrong_input_indexing_map1(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{unexpected input index map for convolution}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0 * 2)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_conv_op_wrong_input_indexing_map2(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{unexpected input index map for convolution}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0 + d1, d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_conv_op_filter_index_map_not_projection(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected output/filter indexing maps to be projected permutations}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d1 + d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @test_conv_op_output_index_map_not_projection(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected output/filter indexing maps to be projected permutations}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0 + d1)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Convolution op illegal if a loop dimension is used to access
// output, filter and is convolved.
func.func @test_conv_op_output_filter_convolved(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{unexpected loop dimension for convolution op}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Convolution op illegal if a loop dimension is used only in the output.
func.func @test_conv_op_output_only_dim(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{unexpected loop dimension for convolution op}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0 + d1)>,
                       affine_map<(d0, d1, d2) -> (d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<reduction>,
                        #test.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?x?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Convolution op illegal if a loop dimension is used only in the filter.
func.func @test_conv_op_filter_only_dim(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{unexpected loop dimension for convolution op}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0 + d1)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<reduction>,
                        #test.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Convolution op illegal if a loop dimension is used only in the input.
func.func @test_conv_op_input_only_dim(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{unexpected loop dimension for convolution op}}
  %0 = test.linalg_conv_op {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0 + d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d1)>,
                       affine_map<(d0, d1, d2) -> (d0)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<reduction>,
                        #test.iterator_type<reduction>]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Convolution op illegal if a loop dimension accessing output is not parallel.
func.func @test_conv_op_non_output_access_loop_parallel(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected all iterators not used to access outputs to be reduction}}
  %0 = test.linalg_conv_op  {
      indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#test.iterator_type<parallel>,
                        #test.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
      ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
         linalg.yield %arg5 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
