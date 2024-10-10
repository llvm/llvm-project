// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @load_number_of_indices(%v : memref<f32>) {
  // expected-error @+2 {{incorrect number of indices for load}}
  %c0 = arith.constant 0 : index
  memref.load %v[%c0] : memref<f32>
}

// -----

func.func @store_number_of_indices(%v : memref<f32>) {
  // expected-error @+3 {{store index operand count not equal to memref rank}}
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  memref.store %f0, %v[%c0] : memref<f32>
}

// -----

func.func @yield_parent(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+1 {{op expected parent op with LinalgOp interface}}
  linalg.yield %arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>
}

// -----

func.func @index_parent() {
  // expected-error @+1 {{op expected parent op with LinalgOp interface}}
  linalg.index 0 : index
}

// -----

func.func @index_dim_lower_than_number_of_loops(%arg0: memref<f32>) {
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

func.func @index_dim_negative(%arg0: memref<f32>) {
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

func.func @generic_no_region(%arg0: memref<f32>) {
  // expected-error @+4 {{expected '{' to begin a region}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> (0)> ],
    iterator_types = []
  } ins(%arg0 : memref<f32>)
}

// -----

func.func @generic_mismatched_num_returns(%arg0: memref<f32>) {
  // expected-error @+6 {{op expected number of yield values (0) to match the number of inits / outs operands of the enclosing LinalgOp (1)}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%0: f32):
      linalg.yield
  }
}

// -----

func.func @generic_wrong_dim_in_map(%arg0: memref<1xi32>) {
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

func.func @generic_wrong_iterator(%arg0: memref<1xi32>) {
  // expected-error @+4 {{unexpected iterator_type (random)}}
  linalg.generic {
    indexing_maps =  [ affine_map<(i) -> (i)> ],
    iterator_types = ["random"]}
      outs(%arg0 : memref<1xi32>) {
    ^bb(%i : i32):
    linalg.yield %i : i32
  }
}

// -----

func.func @generic_one_d_view(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
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

func.func @generic_scalar_view(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  %cst = arith.constant 0.0 : f32
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

func.func @generic_result_0_element_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+7 {{'linalg.yield' op type of yield operand 1 ('i4') doesn't match the element type of the enclosing linalg.generic op ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%0: f32):
      %1 = arith.constant 1: i4
      linalg.yield %1: i4
  }
}

// -----

func.func @generic_singular_maps(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>, %arg1: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
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

func.func @generic_empty_region(%arg0: memref<f32>) {
  %f0 = arith.constant 0.0: f32
  // expected-error @+1 {{op expects region #0 to have 0 or 1 blocks}}
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

func.func @generic_empty_region(%arg0: memref<f32>) {
  %f0 = arith.constant 0.0: f32
  // expected-error @+1 {{op expects to have 1 region with 1 block}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> , affine_map<() -> ()> ],
    iterator_types = []}
    ins(%arg0 : memref<f32>)
   outs(%arg0 : memref<f32>) {
  }
}

// -----

func.func @generic_mismatched_num_arguments(%arg0: memref<f32>) {
  // expected-error @+6 {{'linalg.yield' op expected number of yield values (1) to match the number of inits / outs operands of the enclosing LinalgOp (2)}}
  linalg.generic {
      indexing_maps =  [ affine_map<() -> ()>, affine_map<() -> ()> ],
      iterator_types = []}
      outs(%arg0, %arg0 : memref<f32>, memref<f32>) {
    ^bb(%f: f32):
      linalg.yield %f: f32
  }
}

// -----

func.func @generic_shaped_operand_block_arg_type(%arg0: memref<f32>) {
  // expected-error @+6 {{'linalg.yield' op type of yield operand 1 ('i1') doesn't match the element type of the enclosing linalg.generic op ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> ],
    iterator_types = []}
      outs(%arg0 : memref<f32>) {
    ^bb(%i: i1):
    linalg.yield %i : i1
  }
}

// -----

func.func @generic_scalar_operand_block_arg_type(%arg0: tensor<f32>) {
  // expected-error @+6 {{'linalg.yield' op type of yield operand 1 ('i1') doesn't match the element type of the enclosing linalg.generic op ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> ],
    iterator_types = []}
      outs(%arg0 : tensor<f32>) {
    ^bb(%i: i1):
    linalg.yield %i : i1
  } -> tensor<f32>
}

// -----

func.func @generic_result_0_element_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
  // expected-error @+7 {{type of yield operand 1 ('i1') doesn't match the element type of the enclosing linalg.generic op ('f32')}}
  linalg.generic {
    indexing_maps = [ affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
      outs(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
    ^bb(%i: f32):
      %0 = arith.constant 0: i1
      linalg.yield %0: i1
  }
}

// -----

func.func @generic_result_tensor_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>,
                                 %arg1: tensor<?xf32>) {
  // expected-error @+1 {{expected type of operand #1 ('tensor<?xf32>') to match type of corresponding result ('tensor<f32>')}}
  %0 = linalg.generic {
    indexing_maps = [ affine_map<(i) -> (i)> , affine_map<(i) -> (i)> ],
    iterator_types = ["parallel"]}
       ins(%arg0 : memref<?xf32, affine_map<(i)[off]->(off + i)>>)
      outs(%arg1 : tensor<?xf32>) {
    ^bb(%i: f32, %j: f32):
      linalg.yield %i: f32
  } -> tensor<f32>
}

// -----

func.func @generic(%arg0: memref<?x?xf32>) {
  // expected-error @+6 {{block with no terminator, has %0 = "arith.addf"(%arg1, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32}}
  linalg.generic  {
    indexing_maps = [ affine_map<(i, j) -> (i, j)> ],
    iterator_types = ["parallel", "parallel"]}
      outs(%arg0 : memref<?x?xf32>) {
    ^bb(%0: f32) :
      %1 = arith.addf %0, %0: f32
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

func.func @named_ops(%a3: memref<?x?x?xf32>, %b3: memref<?x?xf32>, %c3: memref<?x?x?xf32>) {
  // expected-error @+1 {{expected operand rank (2) to match the result rank of indexing_map #1 (3)}}
  linalg.batch_matmul ins(%a3, %b3: memref<?x?x?xf32>, memref<?x?xf32>)
                     outs(%c3 : memref<?x?x?xf32>)
  return
}

// -----

func.func @incorrect_region_arg_count(%m: memref<?x?xf32>) {
  // expected-error @+3 {{region expects 3 args, got 2}}
  %res = linalg.matmul ins(%m, %m : memref<?x?xf32>, memref<?x?xf32>)
                       -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return
}

// -----

func.func @matching_inits(%m: memref<?x?xf32>, %t: tensor<?x?xf32>) {
  // expected-error @+1 {{expected type of operand #2 ('tensor<?x?xf32>') to match type of corresponding result ('tensor<?xf32>')}}
  %res = linalg.matmul ins(%m, %m : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%t : tensor<?x?xf32>)
                        -> tensor<?xf32>
  return
}

// -----

func.func @illegal_fill_tensor_no_return(%arg0 : index, %arg1 : index, %arg2 : f32)
{
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  // expected-error @+1 {{expected the number of tensor results (0) to be equal to the number of output tensors (1)}}
  linalg.fill ins(%arg2 : f32) outs(%0 : tensor<?x?xf32>)
}

// -----

func.func @illegal_fill_memref_with_tensor_return
  (%arg0 : memref<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32>
{
  // expected-error @+1 {{expected the number of tensor results (1) to be equal to the number of output tensors (0)}}
  %0 = linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_fill_tensor_with_memref_return
  (%arg0 : tensor<?x?xf32>, %arg1 : f32) -> memref<?x?xf32>
{
  // expected-error @+1 {{result #0 must be variadic of ranked tensor of any type values, but got 'memref<?x?xf32>'}}
  %0 = linalg.fill ins(%arg1 : f32) outs(%arg0 : tensor<?x?xf32>) -> memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func.func @illegal_fill_value_type(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2xf32>) -> tensor<2x2xf32>
{
  // expected-error @+1 {{expected op with scalar input}}
  %0 = linalg.fill ins(%arg1 : tensor<2xf32>) outs(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @invalid_static_matmul(%arg0: memref<2x4xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) {
  // expected-error @+1 {{inferred input/output operand #1 has shape's dimension #0 to be 4, but found 3}}
  linalg.matmul ins(%arg0, %arg1 : memref<2x4xf32>, memref<3x4xf32>)
                      outs(%arg2 :memref<2x4xf32>)
  return
}

// -----

func.func @invalid_indexing_maps_matmul(%arg0: memref<2x4xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) {
  // expected-error @+1 {{expected attribute value}}
  linalg.matmul indexing_maps = [
                       ,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%arg0, %arg1 : memref<2x4xf32>, memref<3x4xf32>)
                      outs(%arg2 :memref<2x4xf32>)
  return
}

// -----

func.func @invalid_matmul_dim_a(%arg0: memref<5x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<5x5xf32>) {
  // expected-error @+1 {{Unexpected dim expression in map result}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5x5xf32>, memref<5x5xf32>) outs(%arg2: memref<5x5xf32>)
  return
}

// -----

func.func @invalid_matmul_dim_b(%arg0: memref<5x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<5x5xf32>) {
  // expected-error @+1 {{Unexpected dim expression in map result}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5x5xf32>, memref<5x5xf32>) outs(%arg2: memref<5x5xf32>)
  return
}

// -----

func.func @invalid_transpose_a_matmul(%lhs: tensor<4x1xf32>, %rhs: tensor<1x64xf32>, %init: tensor<4x64xf32>) -> tensor<4x64xf32> {
  // expected-error @+1 {{inferred input/output operand #1 has shape's dimension #0 to be 4, but found 1}}
  %0 = linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%lhs, %rhs : tensor<4x1xf32>, tensor<1x64xf32>)
                      outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %0: tensor<4x64xf32>
}

// -----

func.func @invalid_transpose_b_matmul(%lhs: tensor<4x1xf32>, %rhs: tensor<1x64xf32>, %init: tensor<4x64xf32>) -> tensor<4x64xf32> {
  // expected-error @+1 {{inferred input/output operand #1 has shape's dimension #1 to be 1, but found 64}}
  %0 = linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
                      ins(%lhs, %rhs : tensor<4x1xf32>, tensor<1x64xf32>)
                      outs(%init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %0: tensor<4x64xf32>
}

// -----

func.func @invalid_bcast_a(%arg0: memref<3xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  // expected-error @+1 {{'linalg.matmul' op Invalid broadcast requested, should be (d2)}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @invalid_bcast_b(%arg0: memref<3x5xf32>, %arg1: memref<7xf32>, %arg2: memref<3x7xf32>) {
  // expected-error @+1 {{'linalg.matmul' op Invalid broadcast requested, should be (d2)}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3x5xf32>, memref<7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @invalid_bcast_a_rank_mismatch(%arg0: memref<3x5xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  // expected-error @+1 {{'linalg.matmul' op expected operand rank (2) to match the result rank of indexing_map #0 (1)}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3x5xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @invalid_bcast_b_rank_mismatch(%arg0: memref<3x5xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  // expected-error @+1 {{'linalg.matmul' op expected operand rank (2) to match the result rank of indexing_map #1 (1)}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3x5xf32>, memref<5x7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @invalid_matmul_bcast_b_transpose_a(%arg0: memref<5x3xf32>, %arg1: memref<7xf32>, %arg2: memref<3x7xf32>) {
  // expected-error @+1 {{inferred input/output operand #1 has shape's dimension #0 to be 5, but found 7}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<5x3xf32>, memref<7xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @invalid_matmul_bcast_b_transpose_a_wrong_dim(%arg0: memref<3x5xf32>, %arg1: memref<5xf32>, %arg2: memref<3x7xf32>) {
  // expected-error @+1 {{'linalg.matmul' op Unexpected dim expression in map result.}}
  linalg.matmul indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                     ]
                     ins(%arg0, %arg1 : memref<3x5xf32>, memref<5xf32>) outs(%arg2: memref<3x7xf32>)
  return
}

// -----

func.func @invalid_indexing_maps_placement_matmul(%lhs: tensor<4x1xf32>, %rhs: tensor<1x64xf32>, %init: tensor<4x64xf32>) {
  // expected-error @+2 {{custom op 'indexing_maps' is unknown (tried 'func.indexing_maps' as well)}}
  linalg.matmul ins(%lhs, %rhs : tensor<4x1xf32>, tensor<1x64xf32>) outs(%init : tensor<4x64xf32>)
                        indexing_maps = [
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>
                      ]
  return
}

// -----

func.func @invalid_static_2d_conv(%input : memref<1x3x4x2xf32>, %filter: memref<3x2x2x1xf32>, %output: memref<1x2x3x1xf32>) {
  // expected-error @+1 {{inferred input/output operand #0 has shape's dimension #1 to be greater than or equal to 4, but found 3}}
  linalg.conv_2d_nhwc_hwcf
    { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter : memref<1x3x4x2xf32>, memref<3x2x2x1xf32>)
    outs(%output : memref<1x2x3x1xf32>)
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

func.func @invalid_reverse(%A: memref<5xf32>, %B: memref<5xf32>) {
  // expected-error @+1 {{unexpected result less than 0 at expression #0 in}}
  linalg.generic #attrs ins(%A: memref<5xf32>) outs(%B: memref<5xf32>) {
                ^bb0(%a: f32, %b: f32):
                linalg.yield %a : f32
        }
        return
}

// -----

func.func @map_binary_wrong_yield_operands(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
   %add = linalg.map
          ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            // expected-error @+1{{'linalg.yield' op expected number of yield values (2) to match the number of inits / outs operands of the enclosing LinalgOp (1)}}
            linalg.yield %0, %0: f32, f32
          }
  func.return %add : tensor<64xf32>
}

// -----

func.func @map_input_mapper_arity_mismatch(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
  // expected-error@+1{{'linalg.map' op expects number of operands to match the arity of mapper, but got: 2 and 3}}
  %add = linalg.map
      ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
      outs(%init:tensor<64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32, %extra_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %0: f32
      }
  func.return %add : tensor<64xf32>
}

// -----

func.func @map_input_mapper_type_mismatch(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
    // expected-error@+1{{'linalg.map' op expected element type of input 'f32' to match bbArg type 'f64'}}
  %add = linalg.map
      ins(%lhs, %rhs : tensor<64xf32>, tensor<64xf32>)
      outs(%init:tensor<64xf32>)
      (%lhs_elem: f64, %rhs_elem: f64) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f64
        linalg.yield %0: f64
      }
  func.return %add : tensor<64xf32>
}

// -----

func.func @map_input_output_shape_mismatch(
    %lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>, %init: tensor<32xf32>)
    -> tensor<32xf32> {
    // expected-error@+1{{'linalg.map' op expected shape of input (64, 64) to match shape of output (32)}}
  %add = linalg.map
      ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%init:tensor<32xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        linalg.yield %0: f32
      }
  func.return %add : tensor<32xf32>
}

// -----

func.func @map_no_operands1() {
  // expected-error @+1 {{'linalg.map' op expected 1 or more operands, but found 0}}
  linalg.map { arith.addf }
}

// -----

func.func @map_no_operands2() {
  // expected-error @+1 {{'linalg.map' op expected 1 or more operands, but found 0}}
  "linalg.map"() ({
    ^bb0:
  }) : () -> ()
}

// -----

func.func @map_no_operands3(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
  // expected-error @+1 {{cannot name an operation with no results}}
  %add = linalg.map { arith.addf }
  func.return %add : tensor<64xf32>
}

// -----

func.func @reduce_input_vs_init_dimension_mismatch(
    %input: tensor<16x32x64xf32>,
    %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  // expected-error @+1 {{'linalg.reduce' op init dimensions [16, 64] doesn't match input dimensions after reduction [16, 32]}}
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [2]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}

// -----

func.func @reduce_dimensions_out_of_range(%input: tensor<16x32x64xf32>,
    %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  // expected-error @+1 {{'linalg.reduce' op dimensions for reduction should be in the range [0, 2].}}
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [3]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}

// -----

func.func @reduce_duplicate_dimensions(%input: tensor<16x32x64xf32>,
    %init: tensor<16xf32>)  -> tensor<16xf32> {
  // expected-error @+1 {{'linalg.reduce' op attribute 'dimensions' failed to satisfy constraint: i64 dense array attribute should be in increasing order}}
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16xf32>)
      dimensions = [1, 1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16xf32>
}

// -----

func.func @reduce_non_increasing_dimensions(%input: tensor<16x32x64xf32>,
    %init: tensor<16xf32>)  -> tensor<16xf32> {
  // expected-error @+1 {{'linalg.reduce' op attribute 'dimensions' failed to satisfy constraint: i64 dense array attribute should be in increasing order}}
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16xf32>)
      dimensions = [2, 1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16xf32>
}

// -----

func.func @reduce_reduced_input_init_rank_mismatch(%input: tensor<16x32x64xf32>,
    %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  // expected-error @+1 {{'linalg.reduce' op number of dimensions after reduction 1 doesn't match the init rank 2}}
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1, 2]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}

// -----

func.func @reduce_wrong_number_of_block_arguments(
    %input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<16x64xf32>)  -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  // expected-error @+1{{'linalg.reduce' op mismatching number of operands and block arguments}}
  %reduce, %reduce2 = linalg.reduce
      ins(%input1, %input2 : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init1, %init2 : tensor<16x64xf32>, tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        linalg.yield %0: f32
      }
  func.return %reduce, %reduce2 : tensor<16x64xf32>, tensor<16x64xf32>
}

// -----

func.func @reduce_wrong_block_argument_input_type(
    %input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<16x64xf32>)  -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  // expected-error @+1{{'linalg.reduce' op input element type 'f32' does not match corresponding block argument type 'f64'}}
  %reduce, %reduce2 = linalg.reduce
      ins(%input1, %input2 : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init1, %init2 : tensor<16x64xf32>, tensor<16x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f64, %out1: f32, %out2: f64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addf %in2, %out2: f64
        linalg.yield %0, %1: f32, f64
      }
  func.return %reduce, %reduce2 : tensor<16x64xf32>, tensor<16x64xf32>
}

// -----

func.func @reduce_wrong_block_argument_output_type(
    %input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<16x64xf64>)  -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  // expected-error @+1{{'linalg.reduce' op output element type 'f64' does not match corresponding block argument type 'f32'}}
  %reduce, %reduce2 = linalg.reduce
      ins(%input1, %input2 : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init1, %init2 : tensor<16x64xf32>, tensor<16x64xf64>)
      dimensions = [1]
      (%in1: f32, %in2: f32, %out1: f32, %out2: f32) {
        %0 = arith.addf %in1, %out1: f32
        linalg.yield %0, %out2: f32, f32
      }
  func.return %reduce, %reduce2 : tensor<16x64xf32>, tensor<16x64xf64>
}

// -----

func.func @reduce_different_input_shapes(%input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<17x32x64xf32>,
    %init2: tensor<17x64xf32>)  -> (tensor<16x64xf32>, tensor<17x64xf32>) {
  // expected-error @+1{{'linalg.reduce' op expects all inputs to have the same shapes. Shape at input-index 1 is not equal to the shape at input-index 0.}}
  %reduce, %reduce2 = linalg.reduce
      ins(%input1, %input2 : tensor<16x32x64xf32>, tensor<17x32x64xf32>)
      outs(%init1, %init2 : tensor<16x64xf32>, tensor<17x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f32, %out1: f32, %out2: f32) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addf %in2, %out2: f32
        linalg.yield %0, %1: f32, f32
      }
  func.return %reduce, %reduce2 : tensor<16x64xf32>, tensor<17x64xf32>
}

// -----

func.func @reduce_different_output_shapes(%input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<17x64xf32>)  -> (tensor<16x64xf32>, tensor<17x64xf32>) {
  // expected-error @+1{{'linalg.reduce' op expects all outputs to have the same shapes. Shape at output-index 1 is not equal to the shape at output-index 0.}}
  %reduce, %reduce2 = linalg.reduce
      ins(%input1, %input2 : tensor<16x32x64xf32>, tensor<16x32x64xf32>)
      outs(%init1, %init2 : tensor<16x64xf32>, tensor<17x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f32, %out1: f32, %out2: f32) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addf %in2, %out2: f32
        linalg.yield %0, %1: f32, f32
      }
  func.return %reduce, %reduce2 : tensor<16x64xf32>, tensor<17x64xf32>
}

// -----

func.func @transpose_invalid_permutation(%input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op permutation is not valid}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 1, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_permutated_dims_mismatch(%input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op dim(result, 0) = 32 doesn't match dim(input, permutation[0]) = 16}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [0, 1, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_rank_permutation_size_mismatch(
    %input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op size of permutation 2 does not match the argument rank 3}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_input_init_rank_mismatch(%input: tensor<16x32xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'linalg.transpose' op input rank 2 does not match init rank 3}}
  %transpose = linalg.transpose
      ins(%input:tensor<16x32xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 0, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_no_operands1() {
  // expected-error @+1 {{'linalg.transpose' op expected 2 operands, but found 0}}
  linalg.transpose permutation = [1, 0, 2]
}

// -----

func.func @transpose_no_operands2() {
  // expected-error @+1 {{'linalg.transpose' op expected 2 operands, but found 0}}
  "linalg.transpose"() <{permutation = array<i64: 1, 0, 2>}> ({
    ^bb0:
  }) : () -> ()
}

// -----

func.func @transpose_no_operands3() -> tensor<32x64x16xf32> {
  // expected-error @+1 {{cannot name an operation with no results}}
  %transpose = linalg.transpose permutation = [1, 0, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @broadcast_input_dims_rank_mismatch(
    %input: tensor<4x16xf32>, %init: tensor<4x8x16xf32>)
    -> tensor<4x8x16xf32> {
  // expected-error @+1 {{'linalg.broadcast' op input rank plus added dimensions does not match init rank. }}
  %bcast = linalg.broadcast
      ins(%input:tensor<4x16xf32>)
      outs(%init:tensor<4x8x16xf32>)
      dimensions = [1, 2]
  func.return %bcast : tensor<4x8x16xf32>
}

// -----

func.func @broadcast_unsorted_dims(
    %input: tensor<4x16xf32>, %init: tensor<4x8x16xf32>)
    -> tensor<4x8x16xf32> {
  // expected-error @+1 {{'linalg.broadcast' op dimension 0 is out of range. expected range: [0, 2], got: 5}}
  %bcast = linalg.broadcast
      ins(%input:tensor<4x16xf32>)
      outs(%init:tensor<4x8x16xf32>)
      dimensions = [5]
  func.return %bcast : tensor<4x8x16xf32>
}

// -----

func.func @broadcast_mapped_dim_mismatch(
    %input: tensor<4x16xf32>, %init: tensor<5x8x16xf32>)
    -> tensor<5x8x16xf32> {
  // expected-error @+1 {{'linalg.broadcast' op input dim 0 should match init dim 0. input: 4, init: 5}}
  %bcast = linalg.broadcast
      ins(%input:tensor<4x16xf32>)
      outs(%init:tensor<5x8x16xf32>)
      dimensions = [1]
  func.return %bcast : tensor<5x8x16xf32>
}

// -----

func.func @broadcast_size_1_extension_not_supported(
    %input: tensor<1x16xf32>, %init: tensor<4x?x16xf32>)
    -> tensor<4x?x16xf32> {
  // expected-error @+1 {{'linalg.broadcast' op input dim 0 should match init dim 0. input: 1, init: 4}}
  %bcast = linalg.broadcast
      ins(%input:tensor<1x16xf32>)
      outs(%init:tensor<4x?x16xf32>)
      dimensions = [1]
  func.return %bcast : tensor<4x?x16xf32>
}

// -----

func.func @broadcast_no_operands1() {
  // expected-error @+1 {{'linalg.broadcast' op expected 2 operands, but found 0}}
  linalg.broadcast dimensions = [1]
}

// -----

func.func @broadcast_no_operands2() {
  // expected-error @+1 {{'linalg.broadcast' op expected 2 operands, but found 0}}
  "linalg.broadcast"() <{dimensions = array<i64: 1>}> ({
    ^bb0:
  }) : () -> ()
}

// -----

func.func @broadcast_no_operands3()
    -> tensor<4x?x16xf32> {
  // expected-error @+1 {{cannot name an operation with no results}}
  %broadcast = linalg.broadcast dimensions = [1]
  func.return %broadcast : tensor<32x64x16xf32>
}

// -----

func.func @missing_iterator_types() {
  // expected-error @below {{expected "iterator_types" array attribute}}
  linalg.generic {} ins() outs()
  return
}

// -----

func.func @illegal_softmax_output_shape(%arg0: tensor<2x16x32xf32>) -> tensor<2x16xf32> {
  %0 = tensor.empty() : tensor<2x16xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>)
                                   outs(%0: tensor<2x16xf32>)
    -> tensor<2x16xf32>
  return %1 : tensor<2x16xf32>
}

// -----

func.func @mmt4d_dims_mismatch(%A: tensor<16x16x8x1xf32>,
                               %B: tensor<16x16x8x1xf32>,
                               %C_in: tensor<16x16x8x1xf32>) -> tensor<16x16x8x1xf32> {
    // expected-error @+1 {{inferred input/output operand #2 has shape's dimension #3 to be 8, but found 1}}
    %res = linalg.mmt4d
                     ins(%A, %B: tensor<16x16x8x1xf32>, tensor<16x16x8x1xf32>)
                     outs(%C_in: tensor<16x16x8x1xf32>)
                     -> tensor<16x16x8x1xf32>
    return %res : tensor<16x16x8x1xf32>
}

// -----

func.func @mmt4d_rank_mismatch(%A: tensor<16x16x8x1xf32>,
                 %B: tensor<16x16x8x1xf32>,
                 %C_in: tensor<8x8xf32>) -> tensor<8x8xf32> {
    // expected-error @+1 {{expected operand rank (2) to match the result rank of indexing_map #2 (4)}}
    %res = linalg.mmt4d
                     ins(%A, %B: tensor<16x16x8x1xf32>, tensor<16x16x8x1xf32>)
                     outs(%C_in: tensor<8x8xf32>)
                     -> tensor<8x8xf32>
    return %res : tensor<8x8xf32>
}

// -----

func.func @mixed_semantics(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: memref<?x?xf32>) {
  // expected-error @+1 {{expected to have pure tensor or buffer semantics}}
  linalg.matmul ins(%a, %b: tensor<?x?xf32>, tensor<?x?xf32>)
               outs(%c: memref<?x?xf32>)
  return
}

// -----

func.func @winograd_filter_transform_height(%arg0: tensor<2x4x3x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  // expected-error @+1 {{expect filter height either equals to r or 1}}
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x4x3x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

// -----

func.func @winograd_filter_transform_width(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  // expected-error @+1 {{expect filter width either equals to r or 1}}
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x3x4x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

// -----

func.func @winograd_filter_transform(%arg0: tensor<2x1x1x5xf32>, %arg1: tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32> {
  // expected-error @+1 {{expect either filter height or width equals to r}}
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<2x1x1x5xf32>) outs(%arg1 : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
  return %0 : tensor<6x6x5x2xf32>
}

// -----

func.func @winograd_filter_dyn(%arg0: tensor<?x3x3x?xf32>, %arg1: tensor<6x5x?x?xf32>) -> tensor<6x5x?x?xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_filter_transform m(4) r(3) ins(%arg0 : tensor<?x3x3x?xf32>) outs(%arg1 : tensor<6x5x?x?xf32>) -> tensor<6x5x?x?xf32>
  return %0 : tensor<6x5x?x?xf32>
}

// -----

func.func @winograd_input_transform_height(%arg0: tensor<2x13x14x5xf32>, %arg1: tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x13x14x5xf32>) outs(%arg1 : tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32>
  return %0 : tensor<6x6x3x3x2x5xf32>
}

// -----

func.func @winograd_input_transform_width(%arg0: tensor<2x14x13x5xf32>, %arg1: tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x14x13x5xf32>) outs(%arg1 : tensor<6x6x3x3x2x5xf32>) -> tensor<6x6x3x3x2x5xf32>
  return %0 : tensor<6x6x3x3x2x5xf32>
}

// -----

func.func @winograd_input_transform_output_tileH(%arg0: tensor<2x14x14x5xf32>, %arg1: tensor<6x6x2x3x2x5xf32>) -> tensor<6x6x2x3x2x5xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x14x14x5xf32>) outs(%arg1 : tensor<6x6x2x3x2x5xf32>) -> tensor<6x6x2x3x2x5xf32>
  return %0 : tensor<6x6x2x3x2x5xf32>
}

// -----

func.func @winograd_input_transform_output_tileW(%arg0: tensor<2x14x14x5xf32>, %arg1: tensor<6x6x3x2x2x5xf32>) -> tensor<6x6x3x2x2x5xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x14x14x5xf32>) outs(%arg1 : tensor<6x6x3x2x2x5xf32>) -> tensor<6x6x3x2x2x5xf32>
  return %0 : tensor<6x6x3x2x2x5xf32>
}

// -----

func.func @winograd_input_transform_output_height(%arg0: tensor<2x14x14x5xf32>, %arg1: tensor<5x6x3x3x2x5xf32>) -> tensor<5x6x3x3x2x5xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x14x14x5xf32>) outs(%arg1 : tensor<5x6x3x3x2x5xf32>) -> tensor<5x6x3x3x2x5xf32>
  return %0 : tensor<5x6x3x3x2x5xf32>
}

// -----

func.func @winograd_input_transform_output_width(%arg0: tensor<2x14x14x5xf32>, %arg1: tensor<6x5x3x3x2x5xf32>) -> tensor<6x5x3x3x2x5xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<2x14x14x5xf32>) outs(%arg1 : tensor<6x5x3x3x2x5xf32>) -> tensor<6x5x3x3x2x5xf32>
  return %0 : tensor<6x5x3x3x2x5xf32>
}

// -----

func.func @winograd_input_dyn(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<6x5x?x?x?x?xf32>) -> tensor<6x5x?x?x?x?xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_input_transform m(4) r(3) ins(%arg0 : tensor<?x?x?x?xf32>) outs(%arg1 : tensor<6x5x?x?x?x?xf32>) -> tensor<6x5x?x?x?x?xf32>
  return %0 : tensor<6x5x?x?x?x?xf32>
}

// -----

func.func @winograd_output_transform_input_height(%arg0: tensor<5x6x3x3x2x2xf32>, %arg1: tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32> {
  // expected-error @+1 {{expect input height equals to input tile size}}
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<5x6x3x3x2x2xf32>) outs(%arg1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
  return %0 : tensor<2x12x12x2xf32>
}

// -----

func.func @winograd_output_transform_input_width(%arg0: tensor<6x5x3x3x2x2xf32>, %arg1: tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32> {
  // expected-error @+1 {{expect input width equals to input tile size}}
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x5x3x3x2x2xf32>) outs(%arg1 : tensor<2x12x12x2xf32>) -> tensor<2x12x12x2xf32>
  return %0 : tensor<2x12x12x2xf32>
}

// -----

func.func @winograd_output_transform_output_height(%arg0: tensor<6x6x3x3x2x2xf32>, %arg1: tensor<2x11x12x2xf32>) -> tensor<2x11x12x2xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x3x3x2x2xf32>) outs(%arg1 : tensor<2x11x12x2xf32>) -> tensor<2x11x12x2xf32>
  return %0 : tensor<2x11x12x2xf32>
}

// -----

func.func @winograd_output_transform_output_width(%arg0: tensor<6x6x3x3x2x2xf32>, %arg1: tensor<2x12x11x2xf32>) -> tensor<2x12x11x2xf32> {
  // expected-error @+1 {{the output shape is not expected}}
  %0 = linalg.winograd_output_transform m(4) r(3) ins(%arg0 : tensor<6x6x3x3x2x2xf32>) outs(%arg1 : tensor<2x12x11x2xf32>) -> tensor<2x12x11x2xf32>
  return %0 : tensor<2x12x11x2xf32>
}
