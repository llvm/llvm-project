// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @load_number_of_indices(%v : memref<f32>) {
  // expected-error @+2 {{incorrect number of indices for load}}
  %c0 = constant 0 : index
  memref.load %v[%c0] : memref<f32>
}

// -----

func @store_number_of_indices(%v : memref<f32>) {
  // expected-error @+3 {{store index operand count not equal to memref rank}}
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  memref.store %f0, %v[%c0] : memref<f32>
}

// -----

func @yield_parent(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+1 {{op expected parent op with LinalgOp interface}}
  linalg.yield %arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>
}

// -----

func @index_parent() {
  // expected-error @+1 {{op expected parent op with LinalgOp interface}}
  linalg.index 0 : index
}

// -----

func @index_dim_lower_than_number_of_loops(%arg0: memref<f32>) {
  // expected-error @+6 {{op expected dim (2) to be lower than the number of loops (0) of the enclosing LinalgOp}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%0: f32):
      linalg.index 2 : index
      linalg.yield %0 : f32
  }
}

// -----

func @index_dim_negative(%arg0: memref<f32>) {
  // expected-error @+6 {{op attribute 'dim' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%0: f32):
      linalg.index -1 : index
      linalg.yield %0 : f32
  }
}

// -----

func @generic_no_region(%arg0: memref<f32>) {
  // expected-error @+5 {{expected '{' to begin a region}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0)> ],
    iterator_types = []
  } ins(%arg0 : memref<f32>)
}

// -----

func @generic_mismatched_num_returns(%arg0: memref<f32>) {
  // expected-error @+6 {{op expected number of yield values (1) to match the number of operands of the enclosing LinalgOp (0)}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%0: f32):
      linalg.yield
  }
}

// -----

func @generic_wrong_dim_in_map(%arg0: memref<1xi32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have 1 dim(s) to match the number of loops}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<1xi32>) {
    ^bb(%i : i32):
    linalg.yield %i : i32
  }
}

// -----

func @generic_one_d_view(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+1 {{expected operand rank (1) to match the result rank of indexing_map #0 (2)}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0, 0)> ],
    iterator_types = []}
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%f : f32):
      linalg.yield %f: f32
  }
}

// -----

func @generic_scalar_view(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  %cst = constant 0.0 : f32
  // expected-error @+1 {{expected operand rank (0) to match the result rank of indexing_map #0 (1)}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0)>, affine_map<() -> (0, 0)> ],
    iterator_types = []}
      ins(%cst : f32)
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%0 : f32, %1 : f32):
      linalg.yield %0: f32
  }
}

// -----

func @generic_result_0_element_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+7 {{'linalg.yield' op type of yield operand 1 ('i4') doesn't match the element type of the enclosing linalg.generic op ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%0: f32):
      %1 = constant 1: i4
      linalg.yield %1: i4
  }
}

// -----

func @generic_singular_maps(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>, %arg1: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+1 {{expected the shape-to-loops map to be non-null}}
  linalg.generic {
    indexing_maps =  [
      affine_map<(i, j) -> (i + j)>,
      affine_map<(i, j) -> (i + j)>
    ],
    iterator_types = ["parallel","parallel"]}
    ins(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>)
   outs(%arg1 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  ^bb(%0: f32, %1: f32):
      linalg.yield %1: f32
  }
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Region tests /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// -----

func @generic_empty_region(%arg0: memref<f32>) {
  %f0 = constant 0.0: f32
  // expected-error @+1 {{op expected 1 region with 1 block}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()>, affine_map<() -> ()> ],
    iterator_types = []}
      ins(%arg0 : memref<f32>)
     outs(%arg0 : memref<f32>) {
    ^bb1:
      linalg.yield %f0: f32
    ^bb2:
      linalg.yield %f0: f32
  }
}

// -----

func @generic_empty_region(%arg0: memref<f32>) {
  %f0 = constant 0.0: f32
  // expected-error @+1 {{linalg.generic' op expected 1 region with 1 block}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> , affine_map<() -> ()> ],
    iterator_types = []}
    ins(%arg0 : memref<f32>)
   outs(%arg0 : memref<f32>) {
  }
}

// -----

func @generic_mismatched_num_arguments(%arg0: memref<f32>) {
  // expected-error @+1 {{expected as many non-induction variable region arguments as the number of input/output operands}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()>, affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0, %arg0 : memref<f32>, memref<f32>) {
    ^bb(%f: f32):
      linalg.yield %f: f32
  }
}

// -----

func @generic_shaped_operand_block_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{expected type of bb argument #0 ('i1') to match element or self type of the corresponding operand ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> ],
    iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%i: i1):
    linalg.yield %i : i1
  }
}

// -----

func @generic_scalar_operand_block_arg_type(%arg0: f32) {
  // expected-error @+1 {{expected type of bb argument #0 ('i1') to match element or self type of the corresponding operand ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> ],
    iterator_types = []}
      outs(%arg0 : f32) {
    ^bb(%i: i1):
    linalg.yield %i : i1
  }
}

// -----

func @generic_result_0_element_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+7 {{type of yield operand 1 ('i1') doesn't match the element type of the enclosing linalg.generic op ('f32')}}
  linalg.generic {
    indexing_maps = [ affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%i: f32):
      %0 = constant 0: i1
      linalg.yield %0: i1
  }
}

// -----

func @generic_result_tensor_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>,
                                 %arg1: tensor<?xf32>) {
  // expected-error @+1 {{expected type of operand #1 ('tensor<?xf32>') to match type of corresponding result ('f32')}}
  %0 = linalg.generic {
    indexing_maps = [ affine_map<(i) -> (i)> , affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
       ins(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>)
      outs(%arg1 : tensor<?xf32>) {
    ^bb(%i: f32, %j: f32):
      linalg.yield %i: f32
  } -> f32
}

// -----

func @generic_result_tensor_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>,
                                 %arg1: tensor<?xf32>) {
  // expected-error @+1 {{unexpected output tensor expression in indexing map #0 a.k.a 'd0' is function of reduction iterator 'd0'}}
  %0 = linalg.generic {
    indexing_maps = [ affine_map<(i) -> (i)> , affine_map<(i) -> (i)> ],
    iterator_types = ["reduction"]}
       ins(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>)
      outs(%arg1 : tensor<?xf32>) {
    ^bb(%i: f32, %j: f32):
      linalg.yield %i: f32
  } -> tensor<?xf32>
}

// -----

func @generic(%arg0: memref<?x?xi4>) {
  // expected-error @+2 {{op expects regions to end with 'linalg.yield', found 'std.addf'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'linalg.yield'}}
  linalg.generic  {
    indexing_maps = [ affine_map<(i, j) -> (i, j)> ],
    iterator_types = ["parallel", "parallel"]}
      outs(%arg0 : memref<?x?xi4>) {
    ^bb(%0: i4) :
      %1 = std.addf %0, %0: i4
  }
  return
}

// -----

// This test is currently disabled: subject to verifier ordering issues.
// Instead, when the ranks are not greater than 2, an assertion will be triggered
// in LinalgStructuredOps.td::ConvOp::iterator_types() for now because the
// verifier inspects the iterator_types. This is slated to become an
// autogenerated op in the future, alleviating the issue.
// func @conv_rank_limit(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
//   // DISABLED_expected -error @+1 {{expects memref ranks to be greater than 2}}
//   linalg.conv(%arg0, %arg1, %arg2) : memref<?xf32>, memref<?xf32>, memref<?xf32>
// }
//
// // -----

// expected-error @+1 {{unknown Linalg type}}
!invalid_type = type !linalg.unknown

// -----

// expected-error @+1 {{expected valid keyword}}
!invalid_type = type !linalg<"?">

// -----

func @named_ops(%a3: memref<?x?x?xf32>, %b3: memref<?x?xf32>, %c3: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected operand rank (2) to match the result rank of indexing_map #1 (3)}}
  linalg.batch_matmul ins(%a3, %b3: memref<?x?x?xf32>, memref<?x?xf32>)
                     outs(%c3 : memref<?x?x?xf32>)
  return
}

// -----

func @incorrect_region_arg_count(%m: memref<?x?xf32>) {
  // expected-error @+3 {{region expects 3 args, got 2}}
  %res = linalg.matmul ins(%m, %m : memref<?x?xf32>, memref<?x?xf32>)
                       -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return
}

// -----

func @matching_inits(%m: memref<?x?xf32>, %t: tensor<?x?xf32>) {
  // expected-error @+1 {{expected type of operand #2 ('tensor<?x?xf32>') to match type of corresponding result ('tensor<?xf32>')}}
  %res = linalg.matmul ins(%m, %m : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%t : tensor<?x?xf32>)
                        -> tensor<?xf32>
  return
}

// -----

func @init_tensor_err(%arg0 : index, %arg1 : index)
{
  // expected-error @+1 {{specified type 'tensor<4x?x?x5xf32>' does not match the inferred type 'tensor<4x5x?x?xf32>'}}
  %1 = linalg.init_tensor [4, 5, %arg0, %arg1] : tensor<4x?x?x5xf32>
  return
}

// -----

func @init_tensor_err(%arg0 : index)
{
  // expected-error @+1 {{expected 4 sizes values}}
  %1 = linalg.init_tensor [4, 5, %arg0] : tensor<4x?x?x5xf32>
  return
}

// -----

func @init_tensor_err(%arg0 : index)
{
  // expected-error @+1 {{expected 2 dynamic sizes values}}
  %1 = "linalg.init_tensor"(%arg0) {static_sizes = [4, -1, -1, 5]} : (index) -> tensor<4x?x?x5xf32>
  return
}

// -----

func @illegal_expanding_reshape_dynamic_tensor
  (%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x4x?xf32>
{
  // expected-error @+1 {{invalid to have a single dimension (2) expanded into multiple dynamic dims (2,4)}}
  %0 = linalg.tensor_expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<?x?x?xf32> into tensor<?x?x?x4x?xf32>
  return %0 : tensor<?x?x?x4x?xf32>
}

// -----


func @illegal_expanding_reshape_static_tensor
  (%arg0: tensor<2x3x20xf32>) -> tensor<2x3x2x4x5xf32>
{
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = linalg.tensor_expand_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<2x3x20xf32> into tensor<2x3x2x4x5xf32>
  return %0 : tensor<2x3x2x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_static_tensor
  (%arg0: tensor<2x3x2x4x5xf32>) -> tensor<2x3x20xf32>
{
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = linalg.tensor_collapse_shape %arg0 [[0], [1], [2, 3, 4]]
      : tensor<2x3x2x4x5xf32> into tensor<2x3x20xf32>
  return %0 : tensor<2x3x20xf32>
}

// -----

func @illegal_expanding_reshape_mixed_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x4x5xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func @illegal_expanding_reshape_mixed_tensor_2(%arg0 : tensor<?x?xf32>) -> tensor<?x4x5xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = linalg.tensor_expand_shape %arg0 [[0], [1, 2]]
      : tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_tensor(%arg0 : tensor<?x4x5xf32>) -> tensor<?x?xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2]]
      : tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_tensor_2(%arg0 : tensor<?x4x5xf32>) -> tensor<?x?xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = linalg.tensor_collapse_shape %arg0 [[0], [1, 2]]
      : tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @pad_result_type(%arg0: tensor<?x2x3x4xi32>, %arg1: index, %arg2: i32) -> tensor<?x?x?x8xf32> {
  // expected-error @+1 {{specified type 'tensor<?x?x?x8xf32>' does not match the inferred type 'tensor<?x?x?x9xi32>}}
  %0 = linalg.pad_tensor %arg0 low[1, %arg1, 2, 2] high[1, 2, %arg1, 3] {
  ^bb0(%arg3: index, %arg4: index):  // no predecessors
    linalg.yield %arg2 : i32
  } : tensor<?x2x3x4xi32> to tensor<?x?x?x8xf32>
  return %0 : tensor<?x?x?x8xf32>
}

// -----

func @pad_number_of_block_args(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{expected the block to have 2 arguments}}
  %0 = linalg.pad_tensor %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):  // no predecessors
    linalg.yield %arg1 : i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @pad_no_block(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{op region #0 ('region') failed to verify constraint: region with 1 blocks}}
  %0 = linalg.pad_tensor %arg0 low[1, 2] high[2, 3] {
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @pad_block_args(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+1 {{op expected block argument 1 to be an index}}
  %0 = linalg.pad_tensor %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
    linalg.yield %arg1 : i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @pad_num_yields(%arg0: tensor<?x4xi32>, %arg1: i32) -> tensor<?x9xi32> {
  // expected-error @+3 {{op expected single yield operand (got 2)}}
  %0 = linalg.pad_tensor %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index):  // no predecessors
    linalg.yield %arg1, %arg1 : i32, i32
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @pad_yield_type(%arg0: tensor<?x4xi32>, %arg1: i8) -> tensor<?x9xi32> {
  // expected-error @+3 {{op expected yield type to match shape element type}}
  %0 = linalg.pad_tensor %arg0 low[1, 2] high[2, 3] {
  ^bb0(%arg2: index, %arg3: index):  // no predecessors
    linalg.yield %arg1 : i8
  } : tensor<?x4xi32> to tensor<?x9xi32>
  return %0 : tensor<?x9xi32>
}

// -----

func @illegal_fill_tensor_no_return(%arg0 : index, %arg1 : index, %arg2 : f32)
{
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  // expected-error @+1 {{expected the number of results (0) to be equal to the number of output tensors (1)}}
  linalg.fill(%arg2, %0) : f32, tensor<?x?xf32>
}

// -----

func @illegal_fill_memref_with_return(%arg0 : memref<?x?xf32>, %arg1 : f32) -> memref<?x?xf32>
{
  // expected-error @+1 {{expected the number of results (1) to be equal to the number of output tensors (0)}}
  %0 = linalg.fill(%arg1, %arg0) : f32, memref<?x?xf32> -> memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func @illegal_fill_memref_with_tensor_return
  (%arg0 : memref<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32>
{
  // expected-error @+1 {{expected the number of results (1) to be equal to the number of output tensors (0)}}
  %0 = linalg.fill(%arg1, %arg0) : f32, memref<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @illegal_fill_tensor_with_memref_return
  (%arg0 : tensor<?x?xf32>, %arg1 : f32) -> memref<?x?xf32>
{
  // expected-error @+1 {{expected type of operand #1 ('tensor<?x?xf32>') to match type of corresponding result ('memref<?x?xf32>')}}
  %0 = linalg.fill(%arg1, %arg0) : f32, tensor<?x?xf32> -> memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func @invalid_static_matmul(%arg0: memref<2x4xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) {
  // expected-error @+1 {{inferred input/output operand #1 has shape's dimension #0 to be 4, but found 3}}
  linalg.matmul ins(%arg0, %arg1 : memref<2x4xf32>, memref<3x4xf32>)
                      outs(%arg2 :memref<2x4xf32>)
  return
}

// -----

func @invalid_static_2d_conv(%input : memref<1x3x4x2xf32>, %filter: memref<3x2x2x1xf32>, %output: memref<1x2x3x1xf32>) {
  // expected-error @+1 {{inferred input/output operand #0 has shape's dimension #1 to be greater than or equal to 4, but found 3}}
  linalg.conv_2d_nhwc_hwcf
    { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : memref<1x3x4x2xf32>, memref<3x2x2x1xf32>)
    outs(%output : memref<1x2x3x1xf32>)
  return
}

// -----

#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func private @foo(%A: memref<192x192xf32>, %B: memref<192x192xf32>,
                  %C: memref<192x192xf32>) -> ()

func @tiled_loop_incorrent_num_yield_operands(%A: memref<192x192xf32>,
    %B: memref<192x192xf32>, %C: memref<192x192xf32>,
    %C_tensor: tensor<192x192xf32>) {
  %c24 = constant 24 : index
  %c0 = constant 0 : index
  %c192 = constant 192 : index
  %0 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c192, %c192)
      step (%c24, %c24)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B: memref<192x192xf32>)
      outs (%CT_ = %C_tensor: tensor<192x192xf32>,
            %C_ = %C: memref<192x192xf32>) {
        call @foo(%A_, %B_, %C_)
          : (memref<192x192xf32>, memref<192x192xf32>, memref<192x192xf32>)-> ()
    // expected-error @+1 {{expected number of tensor output args = 1 to match the number of yield operands = 0}}
    linalg.yield
  }
  return
}

// -----

#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func private @foo(%A: memref<192x192xf32>, %B: memref<192x192xf32>,
                  %C: memref<192x192xf32>) -> tensor<f32>

func @tiled_loop_incorrent_yield_operand_type(%A: memref<192x192xf32>,
    %B: memref<192x192xf32>, %C: memref<192x192xf32>,
    %C_tensor: tensor<192x192xf32>) {
  %c24 = constant 24 : index
  %c0 = constant 0 : index
  %c192 = constant 192 : index
  %0 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c192, %c192)
      step (%c24, %c24)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B: memref<192x192xf32>)
      outs (%CT_ = %C_tensor: tensor<192x192xf32>,
            %C_ = %C: memref<192x192xf32>) {
        %1 = call @foo(%A_, %B_, %C_)
          : (memref<192x192xf32>, memref<192x192xf32>, memref<192x192xf32>)-> tensor<f32>
    // expected-error @+1 {{expected yield operand 0 with type = 'tensor<f32>' to match output arg type = 'tensor<192x192xf32>}}
    linalg.yield %1 : tensor<f32>
  }
  return
}

// -----

func private @foo(%A: memref<192x192xf32>, %B: memref<192x192xf32>,
                  %C: memref<192x192xf32>) -> ()

func @tiled_loop_incorrent_iterator_types_count(%A: memref<192x192xf32>,
    %B: memref<192x192xf32>, %C: memref<192x192xf32>,
    %C_tensor: tensor<192x192xf32>) {
  %c24 = constant 24 : index
  %c0 = constant 0 : index
  %c192 = constant 192 : index
  // expected-error @+1 {{expected iterator types array attribute size = 1 to match the number of loops = 2}}
  %0 = "linalg.tiled_loop"(%c0, %c0, %c192, %c192, %c24, %c24, %A, %B, %C_tensor, %C) ( {
    ^bb0(%arg4: index, %arg5: index, %A_: memref<192x192xf32>,
         %B_: memref<192x192xf32>, %CT_: tensor<192x192xf32>,
         %C_: memref<192x192xf32>):
      call @foo(%A_, %B_, %C_)
          : (memref<192x192xf32>, memref<192x192xf32>, memref<192x192xf32>)-> ()
      linalg.yield %CT_ : tensor<192x192xf32>
    }) {
      iterator_types = ["parallel"],
      operand_segment_sizes = dense<2> : vector<5xi32>
    } : (index, index, index, index, index, index, memref<192x192xf32>,
      memref<192x192xf32>, tensor<192x192xf32>, memref<192x192xf32>
    ) -> tensor<192x192xf32>
  return
}

// -----

func private @foo(%A: memref<100xf32>) -> ()

func @tiled_loop_incorrent_block_arg_type(%A: memref<192xf32>) {
  %c0 = constant 0 : index
  %c192 = constant 192 : index
  %c24 = constant 24 : index
  // expected-error @+1 {{expected output arg 0 with type = 'memref<192xf32>' to match region arg 1 type = 'memref<100xf32>'}}
  "linalg.tiled_loop"(%c0, %c192, %c24, %A) ( {
    ^bb0(%arg4: index, %A_: memref<100xf32>):
      call @foo(%A_) : (memref<100xf32>)-> ()
      linalg.yield
    }) {
      iterator_types = ["parallel"],
      operand_segment_sizes = dense<[1, 1, 1, 0, 1]> : vector<5xi32>
    } : (index, index, index, memref<192xf32>) -> ()
  return
}

// -----

#attrs = {
        indexing_maps = [
                affine_map<(i) -> (3 - i)>,
                affine_map<(i) -> (i)>
        ],
        iterator_types = ["parallel"]
}

func @invalid_reverse(%A: memref<5xf32>, %B: memref<5xf32>) {
  // expected-error @+1 {{unexpected result less than 0 at expression #0 in}}
  linalg.generic #attrs ins(%A: memref<5xf32>) outs(%B: memref<5xf32>) {
                ^bb0(%a: f32, %b: f32):
                linalg.yield %a : f32
        }
        return
}
