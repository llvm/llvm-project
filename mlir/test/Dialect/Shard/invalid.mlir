// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{rank of grid is expected to be a positive integer}}
shard.grid @grid0(shape = [])

// -----

// expected-error@+1 {{custom op 'shard.grid' Failed parsing dimension list. Did you mean an empty list? It must be denoted by "[]".}}
shard.grid @grid0(shape = -1)

// -----

shard.grid @grid0(shape = 2x4)

func.func @grid_axis_duplicated_different_subarray(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{grid axis duplicated}}
  %s = shard.sharding @grid0 split_axes = [[0], [0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @grid_axis_duplicated_same_subarray(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{grid axis duplicated}}
  %s = shard.sharding @grid0 split_axes = [[0, 0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @grid_axis_negtive_in_split_part(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{grid axis is expected to be non-negative}}
  %s = shard.sharding @grid0 split_axes = [[-1]] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

func.func @sharding_attribute_invalid_nested_symbol(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{custom op 'shard.sharding' invalid kind of attribute specified}}
  %s = shard.sharding @a::@b split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

func.func @sharding_attribute_invalid_halo(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{halo sizes must be specified for all split axes}}
  %s = shard.sharding @grid0 split_axes = [[0], [1]] halo_sizes = [1, 2] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

func.func @sharding_attribute_invalid_sizes(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{halo sizes and shard offsets are mutually exclusive}}
  %s = shard.sharding @grid0 split_axes = [[0]] halo_sizes = [1, 2] sharded_dims_offsets = [0, 2, 2] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

shard.grid @grid_dyn(shape = ?x?)
func.func @sharding_dyn_grid_and_sizes(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{sharded dims offsets are not allowed for device grids with dynamic shape}}
  %s = shard.sharding @grid_dyn split_axes = [[0]] sharded_dims_offsets = [0, 2, 2] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

shard.grid @grid0(shape = 2x4)
func.func @sharding_sizes_count(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{sharded dims offsets has wrong size}}
  %s = shard.sharding @grid0 split_axes = [[0], [1]] sharded_dims_offsets = [0, 2, 4, 0, 2, 4, 6] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

shard.grid @grid0(shape = 4)
func.func @sharding_sizes_decreasing(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{sharded dims offsets must be non-decreasing}}
  %s = shard.sharding @grid0 split_axes = [[0]] sharded_dims_offsets = [0, 2, 3, 2] : !shard.sharding
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @grid_shape_grid_axis_out_of_bounds() -> (index, index) {
  // expected-error@+1 {{0-based grid axis index 2 is out of bounds. The referenced grid "grid0" is of rank 2.}}
  %0:2 = shard.grid_shape @grid0 axes = [0, 2] : index, index
  return %0#0, %0#1 : index, index
}

// -----

shard.grid @grid0(shape = 1x2x3)

func.func @grid_shape_duplicate_grid_axis() -> (index, index, index) {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0:3 = shard.grid_shape @grid0 axes = [0, 2, 0] : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @grid_shape_wrong_number_of_results() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 1.}}
  %0:2 = shard.grid_shape @grid0 axes = [0] : index, index
  return %0#0, %0#1 : index, index
}

// -----

shard.grid @grid0(shape = 1x2x3)

func.func @grid_shape_wrong_number_of_results_empty_grid_axes() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 3.}}
  %0:2 = shard.grid_shape @grid0 : index, index
  return %0#0, %0#1 : index, index
}

// -----

func.func @grid_shape_invalid_grid_name() -> (index) {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.grid_shape @this_grid_symbol_does_not_exist : index
  return %0#0 : index
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @process_multi_index_grid_axis_out_of_bounds() -> (index, index) {
  // expected-error@+1 {{0-based grid axis index 2 is out of bounds. The referenced grid "grid0" is of rank 2.}}
  %0:2 = shard.process_multi_index on @grid0 axes = [0, 2] : index, index
  return %0#0, %0#1 : index, index
}

// -----

shard.grid @grid0(shape = 1x2x3)

func.func @process_multi_index_duplicate_grid_axis() -> (index, index, index) {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0:3 = shard.process_multi_index on @grid0 axes = [0, 2, 0] : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @process_multi_index_wrong_number_of_results() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 1.}}
  %0:2 = shard.process_multi_index on @grid0 axes = [0] : index, index
  return %0#0, %0#1 : index, index
}

// -----

shard.grid @grid0(shape = 1x2x3)

func.func @process_multi_index_wrong_number_of_results_empty_grid_axes() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 3.}}
  %0:2 = shard.process_multi_index on @grid0 : index, index
  return %0#0, %0#1 : index, index
}

// -----

func.func @process_multi_index_invalid_grid_name() -> (index) {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.process_multi_index on @this_grid_symbol_does_not_exist : index
  return %0 : index
}

// -----

func.func @process_linear_index_invalid_grid_name() -> (index) {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.process_linear_index on @this_grid_symbol_does_not_exist : index
  return %0 : index
}

// -----

func.func @all_reduce_invalid_grid_symbol(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.all_reduce %arg0 on @this_grid_symbol_does_not_exist reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @all_reduce_invalid_grid_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  // expected-error@+1 {{0-based grid axis index 2 is out of bounds. The referenced grid "grid0" is of rank 2.}}
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [2] reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @all_reduce_duplicate_grid_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0, 1, 0] reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @all_reduce_invalid_tensor_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<5xf64> {
  // expected-error@+1 {{'shard.all_reduce' op requires the same shape for all operands and results}}
  %0 = shard.all_reduce %arg0 on @grid0 : tensor<4xf32> -> tensor<5xf64>
  return %0 : tensor<5xf64>
}

// -----

func.func @all_gather_invalid_grid_symbol(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.all_gather %arg0 on @this_grid_symbol_does_not_exist gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @all_gather_invalid_grid_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{0-based grid axis index 2 is out of bounds. The referenced grid "grid0" is of rank 2.}}
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [2] gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @all_reduce_duplicate_grid_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [2, 2] gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @all_gather_invalid_non_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 4, but got 5.}}
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [0] gather_axis = 0
    : tensor<3x4xf32> -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

shard.grid @grid0(shape = 1x2)

func.func @all_gather_invalid_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 8, but got 5.}}
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [1] gather_axis = 1
    : tensor<3x4xf32> -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @all_gather_invalid_gather_axis_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 3.}}
  %0 = shard.all_gather %arg0 on @grid0 gather_axis = 0
    : tensor<?xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @all_gather_invalid_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis 1 is out of bounds [0, 1).}}
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [0] gather_axis = 1
    : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @all_gather_invalid_negative_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis -1 is out of bounds [0, 1).}}
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [0] gather_axis = -1
    : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @all_slice_duplicate_grid_axis(
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.all_slice %arg0 on @grid0 grid_axes = [0, 0]
    slice_axis = 0
    : tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @all_slice_invalid_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 2.}}
  %0 = shard.all_slice %arg0 on @grid0
    slice_axis = 0
    : tensor<?xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @all_slice_invalid_static_dimension_size(
    %arg0 : tensor<3xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = shard.all_slice %arg0 on @grid0 grid_axes = [0]
    slice_axis = 0
    : tensor<3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @all_slice_invalid_operand_static_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Operand dimension size 4 is not divisible by collective device group size 3 for tensor axis 0.}}
  %0 = shard.all_slice %arg0 on @grid0 grid_axes = [0]
    slice_axis = 0
    : tensor<4xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @all_to_all_invalid_grid_symbol(
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.all_to_all %arg0 on @this_grid_symbol_does_not_exist
    split_axis = 1 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// -----

shard.grid @grid0(shape = 1)

func.func @all_to_all_duplicate_grid_axis(
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.all_to_all %arg0 on @grid0 grid_axes = [0, 0]
    split_axis = 0 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// -----

shard.grid @grid0(shape = ?x1)

func.func @all_to_all_invalid_non_dynamic_result_dimension_induced_by_dynamic_device_group(
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected dynamic, but got 6.}}
  %0 = shard.all_to_all %arg0 on @grid0 grid_axes = [0]
    split_axis = 0 concat_axis = 1
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// -----

shard.grid @grid0(shape = 1x1)

func.func @all_to_all_invalid_non_dynamic_result_split_dimension_induced_by_dynamic_operand_dimension(
    %arg0 : tensor<?x6xi8>) -> tensor<3x?xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 3.}}
  %0 = shard.all_to_all %arg0 on @grid0 grid_axes = [1]
    split_axis = 0 concat_axis = 1
    : tensor<?x6xi8> -> tensor<3x?xi8>
  return %0 : tensor<3x?xi8>
}

// -----

shard.grid @grid0(shape = 1x1)

func.func @all_to_all_invalid_non_dynamic_result_concat_dimension_induced_by_dynamic_operand_dimension(
    %arg0 : tensor<3x?xi8>) -> tensor<?x3xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected dynamic, but got 3.}}
  %0 = shard.all_to_all %arg0 on @grid0 grid_axes = [1]
    split_axis = 0 concat_axis = 1
    : tensor<3x?xi8> -> tensor<?x3xi8>
  return %0 : tensor<?x3xi8>
}

// -----

shard.grid @grid0(shape = 3)

func.func @all_to_all_invalid_non_dynamic_result_concat_dimension_size(
    %arg0 : tensor<3x2xi8>) -> tensor<1x7xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 6, but got 7.}}
  %0 = shard.all_to_all %arg0  on @grid0 grid_axes = [0]
    split_axis = 0 concat_axis = 1
    : tensor<3x2xi8> -> tensor<1x7xi8>
  return %0 : tensor<1x7xi8>
}

// -----

shard.grid @grid0(shape = 3)

func.func @all_to_all_invalid_non_dynamic_result_split_dimension_size(
    %arg0 : tensor<3x2xi8>) -> tensor<2x6xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = shard.all_to_all %arg0 on @grid0 grid_axes = [0]
    split_axis = 0 concat_axis = 1
    : tensor<3x2xi8> -> tensor<2x6xi8>
  return %0 : tensor<2x6xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @broadcast_root_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = shard.broadcast %arg0 on @grid0 grid_axes = [0]
    root = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @broadcast_root_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = shard.broadcast %arg0 on @grid0 grid_axes = [0]
    root = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @broadcast_different_input_and_result_type(
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // expected-error@+1 {{'shard.broadcast' op failed to verify that all of {input, result} have same element type}}
  %0 = shard.broadcast %arg0 on @grid0 grid_axes = [0]
    root = [2]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----

shard.grid @grid0(shape = 1)

func.func @gather_wrong_return_element_type(
    %arg0 : tensor<1xf32>) -> tensor<1xi8> {
  // expected-error@+1 {{'shard.gather' op failed to verify that all of {input, result} have same element type}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0] gather_axis = 0 root = [0]
    : (tensor<1xf32>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

shard.grid @grid0(shape = 1)

func.func @gather_invalid_non_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 4, but got 5.}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0] gather_axis = 0 root = [0]
    : (tensor<3x4xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

shard.grid @grid0(shape = 1x2)

func.func @gather_invalid_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 8, but got 5.}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [1] gather_axis = 1 root = [0]
    : (tensor<3x4xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @gather_invalid_gather_axis_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 3.}}
  %0 = shard.gather %arg0 on @grid0 gather_axis = 0 root = []
    : (tensor<?xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @gather_invalid_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis 1 is out of bounds [0, 1).}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0] gather_axis = 1 root = [0]
    : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

shard.grid @grid0(shape = 1)

func.func @gather_invalid_negative_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis -1 is out of bounds [0, 1).}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0] gather_axis = -1 root = [0]
    : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @gather_root_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<6xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0] gather_axis = 0
    root = [3]
    : (tensor<2xi8>) -> tensor<6xi8>
  return %0 : tensor<6xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @gather_root_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0] gather_axis = 0
    root = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @receive_source_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "source". Got 3, but expected value in the range [0, 2].}}
  %0 = shard.recv %arg0 on @grid0 grid_axes = [0]
    source = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @receive_source_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "source" has unexpected multi-index size 2. Expected 1.}}
  %0 = shard.recv %arg0 on @grid0 grid_axes = [0]
    source = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @receive_different_input_and_result_type(
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // expected-error@+1 {{'shard.recv' op failed to verify that all of {input, result} have same element type}}
  %0 = shard.recv %arg0 on @grid0 grid_axes = [0]
    source = [2]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @reduce_root_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = shard.reduce %arg0 on @grid0 grid_axes = [0]
    root = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @reduce_root_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = shard.reduce %arg0 on @grid0 grid_axes = [0]
    root = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @reduce_different_input_and_result_shape(
    %arg0 : tensor<2xi8>) -> tensor<3xi16> {
  // expected-error@+1 {{'shard.reduce' op failed to verify that all of {input, result} have same shape}}
  %0 = shard.reduce %arg0 on @grid0 grid_axes = [0]
    root = [2]
    : (tensor<2xi8>) -> tensor<3xi16>
  return %0 : tensor<3xi16>
}

// -----

shard.grid @grid0(shape = 3)

func.func @reduce_scatter_duplicate_grid_axis(
    %arg0 : tensor<?xf32>) -> tensor<?xf64> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.reduce_scatter %arg0 on @grid0 grid_axes = [0, 0] scatter_axis = 0
    : tensor<?xf32> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

shard.grid @grid0(shape = 3)

func.func @reduce_scatter_invalid_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<2xf64> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 2.}}
  %0 = shard.reduce_scatter %arg0 on @grid0 scatter_axis = 0
    : tensor<?xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// -----

shard.grid @grid0(shape = 3)

func.func @reduce_scatter_invalid_static_dimension_size(
    %arg0 : tensor<3xf32>) -> tensor<2xf64> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = shard.reduce_scatter %arg0 on @grid0 grid_axes = [0] scatter_axis = 0
    : tensor<3xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// -----

shard.grid @grid0(shape = 3)

func.func @reduce_scatter_invalid_operand_static_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<?xf64> {
  // expected-error@+1 {{Operand dimension size 4 is not divisible by collective device group size 3 for tensor axis 0.}}
  %0 = shard.reduce_scatter %arg0 on @grid0 grid_axes = [0] scatter_axis = 0
    : tensor<4xf32> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

shard.grid @grid0(shape = 3)

func.func @scatter_duplicate_grid_axis(
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [0, 0]
    scatter_axis = 0 root = [0, 0]
    : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @scatter_invalid_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 2.}}
  %0 = shard.scatter %arg0 on @grid0
    scatter_axis = 0 root = []
    : (tensor<?xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @scatter_invalid_static_dimension_size(
    %arg0 : tensor<3xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [0]
    scatter_axis = 0 root = [1]
    : (tensor<3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

shard.grid @grid0(shape = 3)

func.func @scatter_invalid_operand_static_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Operand dimension size 4 is not divisible by collective device group size 3 for tensor axis 0.}}
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [0]
    scatter_axis = 0 root = [1]
    : (tensor<4xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @scatter_root_dimension_out_of_bounds(
    %arg0 : tensor<3xi8>) -> tensor<1xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [0]
    scatter_axis = 0 root = [3]
    : (tensor<3xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @scatter_root_wrong_number_dimensions(
    %arg0 : tensor<3xi8>) -> tensor<1xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [0]
    scatter_axis = 0 root = [2, 2]
    : (tensor<3xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @send_destination_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "destination". Got 3, but expected value in the range [0, 2].}}
  %0 = shard.send %arg0 on @grid0 grid_axes = [0]
    destination = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @send_destination_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "destination" has unexpected multi-index size 2. Expected 1.}}
  %0 = shard.send %arg0 on @grid0 grid_axes = [0]
    destination = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

shard.grid @grid0(shape = 3x?)

func.func @send_different_input_and_result_type(
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // expected-error@+1 {{'shard.send' op failed to verify that all of {input, result} have same element type}}
  %0 = shard.send %arg0 on @grid0 grid_axes = [0]
    destination = [2]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----

func.func @shift_invalid_grid_symbol(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{Undefined required grid symbol "this_grid_symbol_does_not_exist".}}
  %0 = shard.shift %arg0 on @this_grid_symbol_does_not_exist
    shift_axis = 0 offset = -2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @shift_invalid_grid_axis(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{0-based grid axis index 2 is out of bounds. The referenced grid "grid0" is of rank 2.}}
  %0 = shard.shift %arg0 on @grid0 grid_axes = [2]
        shift_axis = 2 offset = -2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @shift_duplicate_grid_axis(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{Grid axes contains duplicate elements.}}
  %0 = shard.shift %arg0 on @grid0 grid_axes = [0, 1, 0]
    shift_axis = 0 offset = -2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @shift_invalid_tensor_dimension_size(
    %arg0 : tensor<4xi8>) -> tensor<5xi8> {
  // expected-error@+1 {{'shard.shift' op requires the same shape for all operands and results}}
  %0 = shard.shift %arg0 on @grid0 grid_axes = [0]
    shift_axis = 0 offset = 2
    : tensor<4xi8> -> tensor<5xi8>
  return %0 : tensor<5xi8>
}

// -----

shard.grid @grid0(shape = 2x4)

func.func @shift_invalid_shift_axis(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{Invalid shift axis 1. It must be one of the grouping grid axes.}}
  %0 = shard.shift %arg0 on @grid0 grid_axes = [0]
    shift_axis = 1 offset = 2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}
