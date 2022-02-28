// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @load_number_of_indices(%v : memref<f32>) {
  // expected-error @+2 {{incorrect number of indices for load}}
  %c0 = arith.constant 0 : index
  memref.load %v[%c0] : memref<f32>
}

// -----

func @store_number_of_indices(%v : memref<f32>) {
  // expected-error @+3 {{store index operand count not equal to memref rank}}
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
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

func @generic_wrong_iterator(%arg0: memref<1xi32>) {
  // expected-error @+1 {{op unexpected iterator_type (random)}}
  linalg.generic {
    indexing_maps =  [ affine_map<(i) -> (i)> ],
    iterator_types = ["random"]}
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

func @generic_result_0_element_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
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

func @generic_empty_region(%arg0: memref<f32>) {
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

func @generic_scalar_operand_block_arg_type(%arg0: tensor<f32>) {
  // expected-error @+1 {{expected type of bb argument #0 ('i1') to match element or self type of the corresponding operand ('f32')}}
  linalg.generic {
    indexing_maps =  [ affine_map<() -> ()> ],
    iterator_types = []}
      outs(%arg0 : tensor<f32>) {
    ^bb(%i: i1):
    linalg.yield %i : i1
  } -> tensor<f32>
}

// -----

func @generic_result_0_element_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>) {
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

func @generic_result_tensor_type(%arg0: memref<?xf32, affine_map<(i)[off]->(off + i)>>,
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
  // expected-error @+2 {{op expects regions to end with 'linalg.yield', found 'arith.addf'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'linalg.yield'}}
  linalg.generic  {
    indexing_maps = [ affine_map<(i, j) -> (i, j)> ],
    iterator_types = ["parallel", "parallel"]}
      outs(%arg0 : memref<?x?xi4>) {
    ^bb(%0: i4) :
      %1 = arith.addf %0, %0: i4
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

func @illegal_fill_tensor_no_return(%arg0 : index, %arg1 : index, %arg2 : f32)
{
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  // expected-error @+1 {{expected the number of results (0) to be equal to the number of output tensors (1)}}
  linalg.fill(%arg2, %0) : f32, tensor<?x?xf32>
}

// -----

func @illegal_fill_memref_with_return(%arg0 : memref<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32>
{
  // expected-error @+1 {{op expected the number of results (1) to be equal to the number of output tensors (0)}}
  %0 = linalg.fill(%arg1, %arg0) : f32, memref<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
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
  // expected-error @+1 {{op result #0 must be ranked tensor of any type values, but got 'memref<?x?xf32>'}}
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
