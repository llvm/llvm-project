// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @test_fill_op_not_linalg_op(%arg0 : f32, %arg1 : tensor<?xf32>)
     -> tensor<?xf32> {
  // expected-error @+1 {{expected a LinalgOp}}
  %0 = "test.fill_op_not_linalg_op"(%arg0, %arg1)
      : (f32, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func.func @test_fill_op_wrong_num_operands(%arg0 : f32, %arg1 : tensor<?xf32>)
     -> tensor<?xf32> {
  // expected-error @+1 {{expected op with 1 input and 1 output}}
  %0 = test.linalg_fill_op {
      indexing_maps = [#map0, #map0, #map1],
      iterator_types = ["parallel"]}
      ins(%arg0, %arg0 : f32, f32) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
         linalg.yield  %arg2 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

#map1 = affine_map<(d0) -> (d0)>
func.func @test_fill_op_non_scalar_input(%arg0 : tensor<?xf32>,
    %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected op with scalar input}}
  %0 = test.linalg_fill_op {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32):
         linalg.yield  %arg2 : f32
      } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
