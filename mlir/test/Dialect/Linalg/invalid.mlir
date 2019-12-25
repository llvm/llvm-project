// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @load_number_of_indices(%v : memref<f32>) {
  // expected-error @+2 {{incorrect number of indices for load}}
  %c0 = constant 0 : index
  load %v[%c0] : memref<f32>
}

// -----

func @slice_number_of_indexings(%arg0: memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+2 {{expected 2 indexings, got 1}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0] : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>, index, memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @slice_rank_vs_range_indices(%arg0: memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+2 {{op expected rank of the view(1) to be the number of ranges(0)}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0, %c0] : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>, index, index, memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @store_number_of_indices(%v : memref<f32>) {
  // expected-error @+3 {{store index operand count not equal to memref rank}}
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  store %f0, %v[%c0] : memref<f32>
}

// -----

func @transpose_not_permutation(%v : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+1 {{expected a permutation map}}
  linalg.transpose %v (i, j) -> (i, i) : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @transpose_bad_rank(%v : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+1 {{expected a permutation map of same rank as the view}}
  linalg.transpose %v (i) -> (i) : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @yield_parent(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected 'linalg.generic' or 'linalg.indexed_generic' parent op}}
  linalg.yield %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @generic_at_least_2_operands(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected 2 or more operands}}
  linalg.generic {
    args_in = 1,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0: memref<f32>
}

// -----

func @generic_exactly_2_views(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected exactly 2 view operands}}
  linalg.generic {
    args_in = 1,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0, %arg0, %arg0: memref<f32>, memref<f32>, memref<f32>
}

// -----

func @generic_undefined_fun(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun attribute to refer to a defined symbol}}
  linalg.generic {
    args_in = 1,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0, %arg0: memref<f32>, memref<f32>
}

// -----

func @foo() { return }

func @generic_mismatched_num_arguments(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun arguments to match number of views}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0: memref<f32>
}

// -----

func @foo(%0: i32) { return }

func @generic_mismatched_num_returns(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun results to match number of output views}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0: memref<f32>
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

func @generic_symbol_in_map(%arg0: memref<i32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have no symbols}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ ()[N] -> (0) ],
    iterator_types = ["parallel"]
  } %arg0: memref<i32>
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

func @generic_wrong_dim_in_map(%arg0: memref<i32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have 1 dim(s) to match the number of loops}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = ["parallel"]
  } %arg0: memref<i32>
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

func @generic_zero_d_view(%arg0: memref<i32>) {
  // expected-error @+1 {{op expected indexing_map #0 to be 0 to match 0-D view: 'memref<i32>'}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (1) ],
    iterator_types = []
  } %arg0: memref<i32>
}

// -----

func @foo(%0: f32) -> f32 { return %0: f32 }

func @generic_one_d_view(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected indexing_map #0 results to match view rank: 'memref<?xf32, (d0)[s0] -> (d0 + s0)>'}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0, 0) ],
    iterator_types = []
  } %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @foo(%0: i32) -> f32 {
  %1 = constant 0.0: f32
  return %1: f32
}

func @generic_fun_arg_0_element_type(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected fun argument 0 of the same type as elemental type 'f32' of view 0}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @foo(%0: f32) -> i4 {
  %1 = constant 1: i4
  return %1: i4
}

func @generic_fun_result_0_element_type(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected fun result 0 of the same type as elemental type 'f32' of view 0}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @foo(%0: f32, %1: f32) -> f32 { return %1: f32 }

func @generic_singular_maps(%arg0: memref<?xf32, (i)[off]->(off + i)>, %arg1: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected the concatenation of maps in indexing_map to be invertible}}
  linalg.generic {
    args_in = 1,
    args_out = 1,
    fun = @foo,
    indexing_maps =  [
      (i, j) -> (i + j) ,
      (i, j) -> (i + j)
    ],
    iterator_types = ["parallel","parallel"]
  } %arg0, %arg1: memref<?xf32, (i)[off]->(off + i)>, memref<?xf32, (i)[off]->(off + i)>
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Region tests /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// -----

func @generic_empty_region(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected region with 1 block}}
  linalg.generic {
    args_in = 1,
    args_out = 1,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0, %arg0 {
    ^bb1:
    ^bb2:
  }: memref<f32>, memref<f32>
}

// -----

func @generic_mismatched_num_arguments(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected number of block arguments to match number of views}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0 {
    ^bb:
  }: memref<f32>
}

// -----

func @generic_block_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected block argument 0 of the same type as elemental type of output view: 'memref<f32>'}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ () -> (0) ],
    iterator_types = []
  } %arg0 {
    ^bb(%i: i1):
  }: memref<f32>
}

// -----

func @indexed_generic_block_arg_count(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected number of block arguments to match number of views + number of loops}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"]
  } %arg0 {
    ^bb(%f: f32):
  }: memref<f32>
}

// -----

func @indexed_generic_block_induction_var_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected block argument 0 to be of IndexType}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"]
  } %arg0 {
    ^bb(%i: f64, %f: f32):
  }: memref<f32>
}

// -----

func @indexed_generic_block_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected block argument 1 of the same type as elemental type of output view: 'memref<f32>'}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"]
  } %arg0 {
    ^bb(%i: index, %f: i1):
  }: memref<f32>
}

// -----

func @foo(%f: f32) -> (f32) {
  return %f : f32
}
func @indexed_generic_fun_arg_count(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun arguments to match number of views + number of loops}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"],
    fun = @foo
  } %arg0:  memref<f32>
}

// -----

func @foo(%i: i32, %val: f32) -> (f32) {
  return %val : f32
}
func @indexed_generic_fun_induction_var_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun argument 0 to be of IndexType}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    iterator_types = ["parallel"],
    indexing_maps = [ (i) -> (i) ],
    fun = @foo
  } %arg0 : memref<f32>
}

// -----

func @foo(%i: index, %val: i1) -> (i1) {
  return %val : i1
}
func @indexed_generic_fun_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun argument 1 of the same type as elemental type 'f32' of view 0}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"],
    fun = @foo
  } %arg0: memref<f32>
}

// -----

func @foo(%i: index, %val: i1) -> (i1, i1) {
  return %val, %val : i1, i1
}
func @indexed_generic_fun_result_count(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun results to match number of output views}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"],
    fun = @foo
  } %arg0: memref<f32>
}

// -----

func @foo(%i: index, %val: i32) -> (f32) {
  %val_float = sitofp %val : i32 to f32
  return %val_float : f32
}
func @indexed_generic_fun_result_count(%arg0: memref<i32>) {
  // expected-error @+1 {{op expected fun result 0 of the same type as elemental type 'i32' of view 0}}
  linalg.indexed_generic {
    args_in = 0,
    args_out = 1,
    indexing_maps =  [ (d0) -> (d0) ],
    iterator_types = ["parallel"],
    fun = @foo
  } %arg0: memref<i32>
}

// -----

func @generic_fun_result_0_element_type(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+9 {{type of return operand 0 ('i1') doesn't match view element type ('f32')}}
  linalg.generic {
    args_in = 0,
    args_out = 1,
    indexing_maps = [ (i) -> (i) ],
    iterator_types = ["parallel"]
  } %arg0 {
    ^bb(%i: f32):
      %0 = constant 0: i1
      linalg.yield %0: i1
  }: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @generic_fun_result_0_element_type(%arg0: memref<?xf32>) {
  // expected-error @+1 {{'linalg.dot' op expected 3 or more operands}}
  linalg.dot(%arg0, %arg0): memref<?xf32>, memref<?xf32>
}

// -----

// expected-error @+1 {{unknown Linalg type}}
!invalid_type = type !linalg.unknown

// -----

// expected-error @+1 {{expected valid keyword}}
!invalid_type = type !linalg<"?">
