// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @load_number_of_indices(%v : memref<f32>) {
  // expected-error @+2 {{incorrect number of indices for load}}
  %c0 = constant 0 : index
  load %v[%c0] : memref<f32>
}

// -----

func @slice_number_of_indexings(%arg0: memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+2 {{expected 2 indexings, got 1}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0] : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>, index, memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>
}

// -----

func @slice_rank_vs_range_indices(%arg0: memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>) {
  // expected-error @+2 {{op expected rank of the view(1) to be the number of ranges(0)}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0, %c0] : memref<?x?xf32, affine_map<(i, j)[off, M]->(off + M * i + j)>>, index, index, memref<?xf32, affine_map<(i)[off]->(off + i)>>
}

// -----

func @store_number_of_indices(%v : memref<f32>) {
  // expected-error @+3 {{store index operand count not equal to memref rank}}
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  store %f0, %v[%c0] : memref<f32>
}

// -----

func @yield_parent(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+1 {{op expected parent op with LinalgOp interface}}
  linalg.yield %arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>
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
  // expected-error @+1 {{expected shaped value rank (1) to match the result rank of indexing_map #0 (2)}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0, 0)> ],
    iterator_types = []}
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%f : f32):
      linalg.yield %f: f32
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
  // expected-error @+1 {{op expects region #0 to have 0 or 1 blocks}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0)> ],
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
  // expected-error @+1 {{expected as many non-induction variable region arguments as the number of shaped operands}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()>, affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0, %arg0 : memref<f32>, memref<f32>) {
    ^bb(%f: f32):
      linalg.yield %f: f32
  }
}

// -----

func @generic_block_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{expected type of bb argument #0 ('i1') to match element type of corresponding shaped operand ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> ],
    iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%i: i1):
    linalg.yield %i : i1
  }
}

// -----

func @indexed_generic_block_arg_count(%arg0: memref<?xf32>) {
  // expected-error @+1 {{expected as many non-induction variable region arguments as the number of shaped operands}}
  linalg.indexed_generic {
    indexing_maps =  [ affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32>) {
    ^bb(%f: f32):
      linalg.yield %f : f32
  }
}

// -----

func @indexed_generic_block_induction_var_arg_type(%arg0: memref<?xf32>) {
  // expected-error @+1 {{op expected index block argument #0}}
  linalg.indexed_generic {
    indexing_maps =  [ affine_map<(d0) -> (d0)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32>) {
    ^bb(%i: f64, %f: f32):
    linalg.yield %f: f32
  }
}

// -----

func @indexed_generic_block_arg_type(%arg0: memref<?xf32>) {
  // expected-error @+1 {{expected type of bb argument #1 ('i1') to match element type of corresponding shaped operand ('f32')}}
  linalg.indexed_generic {
    indexing_maps =  [ affine_map<(d0) -> (d0)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32>) {
    ^bb(%i: index, %f: i1):
    linalg.yield %i: index
  }
}

// -----

func @indexed_generic_arg_count(%arg0: memref<f32>) {
  // expected-error @+1 {{expected as many non-induction variable region arguments as the number of shaped operands}}
  linalg.indexed_generic {
    indexing_maps =  [ affine_map<()[] -> ()> ],
    iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%0: index, %1: f32):
      linalg.yield %1: f32
  }
  return
}

// -----

func @indexed_generic_result_count(%arg0: memref<?xf32>) {
  // expected-error @+6 {{op expected number of yield values (1) to match the number of operands of the enclosing LinalgOp (2)}}
  linalg.indexed_generic {
    indexing_maps =  [ affine_map<(d0) -> (d0)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32>) {
    ^bb(%i: index, %val: f32):
      linalg.yield %val, %val: f32, f32
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
    indexing_maps = [ affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
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

func @reshape(%arg0: memref<f32>) {
  // expected-error @+1 {{expected non-zero memref ranks}}
  %0 = linalg.reshape %arg0 [affine_map<()->(0)>] : memref<f32> into memref<f32>
}

// -----

func @reshape(%arg0: memref<?xf32>) {
  // expected-error @+1 {{expected to collapse or expand dims}}
  %0 = linalg.reshape %arg0 [affine_map<(i)->(i)>] : memref<?xf32> into memref<?xf32>
}

// -----

func @reshape(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected rank of the collapsed type(2) to be the number of reassociation maps(1)}}
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i, j)>] :
    memref<?x?x?xf32> into memref<?x?xf32, offset: 0, strides: [?, 1]>
}

// -----

func @reshape(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected reassociation map #0 of same rank as expanded memref(3), but got 1}}
  %0 = linalg.reshape %arg0 [affine_map<(i) -> (i)>, affine_map<(i, j, k) -> (k)>] :
    memref<?x?x?xf32> into memref<?x?xf32, offset: 0, strides: [?, 1]>
}

// -----

func @reshape(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected reassociation map #1 to be valid and contiguous}}
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i, j)>, affine_map<(i, j, k) -> (k, j)>] :
    memref<?x?x?xf32> into memref<?x?xf32, offset: 0, strides: [?, 1]>
}

// -----

func @reshape(%arg0: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected collapsed type to be 'memref<?x?xf32>', but got 'memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>'}}
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k) -> (i, j)>, affine_map<(i, j, k) -> (k)>] :
    memref<?x?x?xf32> into memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>>
}

// -----

func @pooling_rank_mismatch(%arg0: memref<?x?x?xf32>,
                            %arg1: memref<2x3xf32>,
                            %arg2: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected shaped value rank (2) to match the result rank of indexing_map #1 (3)}}
  linalg.pooling_max(%arg0, %arg1, %arg2) {strides = [2, 1, 2]}:
    memref<?x?x?xf32>, memref<2x3xf32>, memref<?x?x?xf32>
  return
}

// -----

func @named_ops(%a3: memref<?x?x?xf32>, %b3: memref<?x?xf32>, %c3: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected shaped value rank (2) to match the result rank of indexing_map #1 (3)}}
  linalg.batch_matmul ins(%a3, %b3: memref<?x?x?xf32>, memref<?x?xf32>)
                     outs(%c3 : memref<?x?x?xf32>)
  return
}

// -----

func @incorrect_region_arg_count(%m: memref<?x?xf32>) {
  // expected-error @+3 {{region expects 3 args, got 2}}
  %res = linalg.matmul ins(%m, %m : memref<?x?xf32>, memref<?x?xf32>)
                       -> tensor<?x?xf32>, tensor<?x?xf32>
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
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    tensor<?x?x?xf32> into tensor<?x?x?x4x?xf32>
  return %0 : tensor<?x?x?x4x?xf32>
}

// -----

func @illegal_expanding_reshape_dynamic_memref
  (%arg0: memref<?x?x?xf32>) -> memref<?x?x?x4x?xf32>
{
  // expected-error @+1 {{invalid to have a single dimension (2) expanded into multiple dynamic dims (2,4)}}
  %0 = linalg.reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    memref<?x?x?xf32> into memref<?x?x?x4x?xf32>
  return %0 : memref<?x?x?x4x?xf32>
}

// -----

func @illegal_expanding_reshape_static_tensor
  (%arg0: tensor<2x3x20xf32>) -> tensor<2x3x2x4x5xf32>
{
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    tensor<2x3x20xf32> into tensor<2x3x2x4x5xf32>
  return %0 : tensor<2x3x2x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_static_tensor
  (%arg0: tensor<2x3x2x4x5xf32>) -> tensor<2x3x20xf32>
{
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    tensor<2x3x2x4x5xf32> into tensor<2x3x20xf32>
  return %0 : tensor<2x3x20xf32>
}

// -----

func @illegal_expanding_reshape_static_memref
  (%arg0: memref<2x3x20xf32>) -> memref<2x3x2x4x5xf32>
{
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = linalg.reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    memref<2x3x20xf32> into memref<2x3x2x4x5xf32>
  return %0 : memref<2x3x2x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_static_memref
  (%arg0: memref<2x3x2x4x5xf32>) -> memref<2x3x20xf32>
{
  // expected-error @+1 {{expected dimension 2 of collapsed type to be static value of 40}}
  %0 = linalg.reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d1)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>] :
    memref<2x3x2x4x5xf32> into memref<2x3x20xf32>
  return %0 : memref<2x3x20xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x4x5xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_tensor_2(%arg0 : tensor<?x?xf32>) -> tensor<?x4x5xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>] :
       tensor<?x?xf32> into tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// -----

func @illegal_expanding_reshape_mixed_tensor(%arg0 : tensor<?x4x5xf32>) -> tensor<?x?xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @illegal_expanding_reshape_mixed_tensor_2(%arg0 : tensor<?x4x5xf32>) -> tensor<?x?xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = linalg.tensor_reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>] :
       tensor<?x4x5xf32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_memref(%arg0 : memref<?x?xf32>) -> memref<?x4x5xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<?x?xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func @illegal_collapsing_reshape_mixed_memref_2(%arg0 : memref<?x?xf32>) -> memref<?x4x5xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>] :
       memref<?x?xf32> into memref<?x4x5xf32>
  return %0 : memref<?x4x5xf32>
}

// -----

func @illegal_expanding_reshape_mixed_memref(%arg0 : memref<?x4x5xf32>) -> memref<?x?xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 5}}
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>] :
       memref<?x4x5xf32> into memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func @illegal_expanding_reshape_mixed_memref_2(%arg0 : memref<?x4x5xf32>) -> memref<?x?xf32>
{
  // expected-error @+1 {{expected dimension 1 of collapsed type to be static value of 20}}
  %0 = linalg.reshape %arg0
         [affine_map<(d0, d1, d2) -> (d0)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>] :
       memref<?x4x5xf32> into memref<?x?xf32>
  return %0 : memref<?x?xf32>
}
