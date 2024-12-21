// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{rank of mesh is expected to be a positive integer}}
mesh.mesh @mesh0(shape = [])

// -----

// expected-error@+1 {{custom op 'mesh.mesh' Failed parsing dimension list. Did you mean an empty list? It must be denoted by "[]".}}
mesh.mesh @mesh0(shape = -1)

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_axis_duplicated_different_subarray(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{mesh axis duplicated}}
  %s = mesh.sharding @mesh0 split_axes = [[0], [0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_axis_duplicated_same_subarray(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{mesh axis duplicated}}
  %s = mesh.sharding @mesh0 split_axes = [[0, 0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_axis_duplicated_bewteen_split_and_partial(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{mesh axis duplicated}}
  %s = mesh.sharding @mesh0 split_axes = [[0]] partial=max[0] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_axis_negtive_in_split_part(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{mesh axis is expected to be non-negative}}
  %s = mesh.sharding @mesh0 split_axes = [[-1]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_axis_negtive_in_partial(
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{mesh axis is expected to be non-negative}}
  %s = mesh.sharding @mesh0 split_axes = [[0]] partial=max[-1] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

func.func @sharding_attribute_invalid_nested_symbol(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{custom op 'mesh.sharding' invalid kind of attribute specified}}
  %s = mesh.sharding @a::@b split_axes = [[0]] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

func.func @sharding_attribute_invalid_halo(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{halo sizes must be specified for all split axes}}
  %s = mesh.sharding @mesh0 split_axes = [[0], [1]] halo_sizes = [1, 2] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

func.func @sharding_attribute_invalid_sizes(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{halo sizes and shard offsets are mutually exclusive}}
  %s = mesh.sharding @mesh0 split_axes = [[0]] halo_sizes = [1, 2] sharded_dims_offsets = [0, 2, 2] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

mesh.mesh @mesh_dyn(shape = ?x?)
func.func @sharding_dyn_mesh_and_sizes(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{sharded dims offsets are not allowed for devices meshes with dynamic shape}}
  %s = mesh.sharding @mesh_dyn split_axes = [[0]] sharded_dims_offsets = [0, 2, 2] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

mesh.mesh @mesh0(shape = 2x4)
func.func @sharding_sizes_count(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{sharded dims offsets has wrong size}}
  %s = mesh.sharding @mesh0 split_axes = [[0], [1]] sharded_dims_offsets = [0, 2, 4, 0, 2, 4, 6] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

mesh.mesh @mesh0(shape = 4)
func.func @sharding_sizes_decreasing(%arg0 : tensor<4x8xf32>) {
  // expected-error@+1 {{sharded dims offsets must be non-decreasing}}
  %s = mesh.sharding @mesh0 split_axes = [[0]] sharded_dims_offsets = [0, 2, 3, 2] : !mesh.sharding
  %0 = mesh.shard %arg0 to %s : tensor<4x8xf32>
  return
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_shape_mesh_axis_out_of_bounds() -> (index, index) {
  // expected-error@+1 {{0-based mesh axis index 2 is out of bounds. The referenced mesh "mesh0" is of rank 2.}}
  %0:2 = mesh.mesh_shape @mesh0 axes = [0, 2] : index, index
  return %0#0, %0#1 : index, index
}

// -----

mesh.mesh @mesh0(shape = 1x2x3)

func.func @mesh_shape_duplicate_mesh_axis() -> (index, index, index) {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0:3 = mesh.mesh_shape @mesh0 axes = [0, 2, 0] : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @mesh_shape_wrong_number_of_results() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 1.}}
  %0:2 = mesh.mesh_shape @mesh0 axes = [0] : index, index
  return %0#0, %0#1 : index, index
}

// -----

mesh.mesh @mesh0(shape = 1x2x3)

func.func @mesh_shape_wrong_number_of_results_empty_mesh_axes() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 3.}}
  %0:2 = mesh.mesh_shape @mesh0 : index, index
  return %0#0, %0#1 : index, index
}

// -----

func.func @mesh_shape_invalid_mesh_name() -> (index) {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.mesh_shape @this_mesh_symbol_does_not_exist : index
  return %0#0 : index
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @process_multi_index_mesh_axis_out_of_bounds() -> (index, index) {
  // expected-error@+1 {{0-based mesh axis index 2 is out of bounds. The referenced mesh "mesh0" is of rank 2.}}
  %0:2 = mesh.process_multi_index on @mesh0 axes = [0, 2] : index, index
  return %0#0, %0#1 : index, index
}

// -----

mesh.mesh @mesh0(shape = 1x2x3)

func.func @process_multi_index_duplicate_mesh_axis() -> (index, index, index) {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0:3 = mesh.process_multi_index on @mesh0 axes = [0, 2, 0] : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @process_multi_index_wrong_number_of_results() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 1.}}
  %0:2 = mesh.process_multi_index on @mesh0 axes = [0] : index, index
  return %0#0, %0#1 : index, index
}

// -----

mesh.mesh @mesh0(shape = 1x2x3)

func.func @process_multi_index_wrong_number_of_results_empty_mesh_axes() -> (index, index) {
  // expected-error@+1 {{Unexpected number of results 2. Expected 3.}}
  %0:2 = mesh.process_multi_index on @mesh0 : index, index
  return %0#0, %0#1 : index, index
}

// -----

func.func @process_multi_index_invalid_mesh_name() -> (index) {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.process_multi_index on @this_mesh_symbol_does_not_exist : index
  return %0 : index
}

// -----

func.func @process_linear_index_invalid_mesh_name() -> (index) {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.process_linear_index on @this_mesh_symbol_does_not_exist : index
  return %0 : index
}

// -----

func.func @all_reduce_invalid_mesh_symbol(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.all_reduce %arg0 on @this_mesh_symbol_does_not_exist reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @all_reduce_invalid_mesh_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  // expected-error@+1 {{0-based mesh axis index 2 is out of bounds. The referenced mesh "mesh0" is of rank 2.}}
  %0 = mesh.all_reduce %arg0 on @mesh0 mesh_axes = [2] reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @all_reduce_duplicate_mesh_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.all_reduce %arg0 on @mesh0 mesh_axes = [0, 1, 0] reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @all_reduce_invalid_tensor_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<5xf64> {
  // expected-error@+1 {{'mesh.all_reduce' op requires the same shape for all operands and results}}
  %0 = mesh.all_reduce %arg0 on @mesh0 : tensor<4xf32> -> tensor<5xf64>
  return %0 : tensor<5xf64>
}

// -----

func.func @all_gather_invalid_mesh_symbol(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.all_gather %arg0 on @this_mesh_symbol_does_not_exist gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @all_gather_invalid_mesh_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{0-based mesh axis index 2 is out of bounds. The referenced mesh "mesh0" is of rank 2.}}
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [2] gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @all_reduce_duplicate_mesh_axis(
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [2, 2] gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @all_gather_invalid_non_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 4, but got 5.}}
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 0
    : tensor<3x4xf32> -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

mesh.mesh @mesh0(shape = 1x2)

func.func @all_gather_invalid_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 8, but got 5.}}
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [1] gather_axis = 1
    : tensor<3x4xf32> -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @all_gather_invalid_gather_axis_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 3.}}
  %0 = mesh.all_gather %arg0 on @mesh0 gather_axis = 0
    : tensor<?xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @all_gather_invalid_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis 1 is out of bounds [0, 1).}}
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 1
    : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @all_gather_invalid_negative_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis -1 is out of bounds [0, 1).}}
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = -1
    : tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @all_slice_duplicate_mesh_axis(
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.all_slice %arg0 on @mesh0 mesh_axes = [0, 0]
    slice_axis = 0
    : tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @all_slice_invalid_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 2.}}
  %0 = mesh.all_slice %arg0 on @mesh0
    slice_axis = 0
    : tensor<?xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @all_slice_invalid_static_dimension_size(
    %arg0 : tensor<3xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = mesh.all_slice %arg0 on @mesh0 mesh_axes = [0]
    slice_axis = 0
    : tensor<3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @all_slice_invalid_operand_static_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Operand dimension size 4 is not divisible by collective device group size 3 for tensor axis 0.}}
  %0 = mesh.all_slice %arg0 on @mesh0 mesh_axes = [0]
    slice_axis = 0
    : tensor<4xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @all_to_all_invalid_mesh_symbol(
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.all_to_all %arg0 on @this_mesh_symbol_does_not_exist
    split_axis = 1 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @all_to_all_duplicate_mesh_axis(
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.all_to_all %arg0 on @mesh0 mesh_axes = [0, 0]
    split_axis = 0 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// -----

mesh.mesh @mesh0(shape = ?x1)

func.func @all_to_all_invalid_non_dynamic_result_dimension_induced_by_dynamic_device_group(
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected dynamic, but got 6.}}
  %0 = mesh.all_to_all %arg0 on @mesh0 mesh_axes = [0]
    split_axis = 0 concat_axis = 1
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// -----

mesh.mesh @mesh0(shape = 1x1)

func.func @all_to_all_invalid_non_dynamic_result_split_dimension_induced_by_dynamic_operand_dimension(
    %arg0 : tensor<?x6xi8>) -> tensor<3x?xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 3.}}
  %0 = mesh.all_to_all %arg0 on @mesh0 mesh_axes = [1]
    split_axis = 0 concat_axis = 1
    : tensor<?x6xi8> -> tensor<3x?xi8>
  return %0 : tensor<3x?xi8>
}

// -----

mesh.mesh @mesh0(shape = 1x1)

func.func @all_to_all_invalid_non_dynamic_result_concat_dimension_induced_by_dynamic_operand_dimension(
    %arg0 : tensor<3x?xi8>) -> tensor<?x3xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected dynamic, but got 3.}}
  %0 = mesh.all_to_all %arg0 on @mesh0 mesh_axes = [1]
    split_axis = 0 concat_axis = 1
    : tensor<3x?xi8> -> tensor<?x3xi8>
  return %0 : tensor<?x3xi8>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @all_to_all_invalid_non_dynamic_result_concat_dimension_size(
    %arg0 : tensor<3x2xi8>) -> tensor<1x7xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 6, but got 7.}}
  %0 = mesh.all_to_all %arg0  on @mesh0 mesh_axes = [0]
    split_axis = 0 concat_axis = 1
    : tensor<3x2xi8> -> tensor<1x7xi8>
  return %0 : tensor<1x7xi8>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @all_to_all_invalid_non_dynamic_result_split_dimension_size(
    %arg0 : tensor<3x2xi8>) -> tensor<2x6xi8> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = mesh.all_to_all %arg0 on @mesh0 mesh_axes = [0]
    split_axis = 0 concat_axis = 1
    : tensor<3x2xi8> -> tensor<2x6xi8>
  return %0 : tensor<2x6xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @broadcast_root_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = mesh.broadcast %arg0 on @mesh0 mesh_axes = [0]
    root = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @broadcast_root_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = mesh.broadcast %arg0 on @mesh0 mesh_axes = [0]
    root = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @broadcast_different_input_and_result_type(
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // expected-error@+1 {{'mesh.broadcast' op failed to verify that all of {input, result} have same element type}}
  %0 = mesh.broadcast %arg0 on @mesh0 mesh_axes = [0]
    root = [2]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @gather_wrong_return_element_type(
    %arg0 : tensor<1xf32>) -> tensor<1xi8> {
  // expected-error@+1 {{'mesh.gather' op failed to verify that all of {input, result} have same element type}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 0 root = [0]
    : (tensor<1xf32>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @gather_invalid_non_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 4, but got 5.}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 0 root = [0]
    : (tensor<3x4xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

mesh.mesh @mesh0(shape = 1x2)

func.func @gather_invalid_gather_axis_dimension_size(
    %arg0 : tensor<3x4xf32>) -> tensor<3x5xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 1. Expected 8, but got 5.}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [1] gather_axis = 1 root = [0]
    : (tensor<3x4xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @gather_invalid_gather_axis_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 3.}}
  %0 = mesh.gather %arg0 on @mesh0 gather_axis = 0 root = []
    : (tensor<?xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @gather_invalid_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis 1 is out of bounds [0, 1).}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 1 root = [0]
    : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

mesh.mesh @mesh0(shape = 1)

func.func @gather_invalid_negative_gather_axis(
    %arg0 : tensor<3xf32>) -> tensor<3xf32> {
  // expected-error@+1 {{Gather axis -1 is out of bounds [0, 1).}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = -1 root = [0]
    : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @gather_root_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<6xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 0
    root = [3]
    : (tensor<2xi8>) -> tensor<6xi8>
  return %0 : tensor<6xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @gather_root_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0] gather_axis = 0
    root = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @receive_source_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "source". Got 3, but expected value in the range [0, 2].}}
  %0 = mesh.recv %arg0 on @mesh0 mesh_axes = [0]
    source = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @receive_source_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "source" has unexpected multi-index size 2. Expected 1.}}
  %0 = mesh.recv %arg0 on @mesh0 mesh_axes = [0]
    source = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @receive_different_input_and_result_type(
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // expected-error@+1 {{'mesh.recv' op failed to verify that all of {input, result} have same element type}}
  %0 = mesh.recv %arg0 on @mesh0 mesh_axes = [0]
    source = [2]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @reduce_root_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = mesh.reduce %arg0 on @mesh0 mesh_axes = [0]
    root = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @reduce_root_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = mesh.reduce %arg0 on @mesh0 mesh_axes = [0]
    root = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @reduce_different_input_and_result_shape(
    %arg0 : tensor<2xi8>) -> tensor<3xi16> {
  // expected-error@+1 {{'mesh.reduce' op failed to verify that all of {input, result} have same shape}}
  %0 = mesh.reduce %arg0 on @mesh0 mesh_axes = [0]
    root = [2]
    : (tensor<2xi8>) -> tensor<3xi16>
  return %0 : tensor<3xi16>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @reduce_scatter_duplicate_mesh_axis(
    %arg0 : tensor<?xf32>) -> tensor<?xf64> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.reduce_scatter %arg0 on @mesh0 mesh_axes = [0, 0] scatter_axis = 0
    : tensor<?xf32> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @reduce_scatter_invalid_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<2xf64> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 2.}}
  %0 = mesh.reduce_scatter %arg0 on @mesh0 scatter_axis = 0
    : tensor<?xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @reduce_scatter_invalid_static_dimension_size(
    %arg0 : tensor<3xf32>) -> tensor<2xf64> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = mesh.reduce_scatter %arg0 on @mesh0 mesh_axes = [0] scatter_axis = 0
    : tensor<3xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @reduce_scatter_invalid_operand_static_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<?xf64> {
  // expected-error@+1 {{Operand dimension size 4 is not divisible by collective device group size 3 for tensor axis 0.}}
  %0 = mesh.reduce_scatter %arg0 on @mesh0 mesh_axes = [0] scatter_axis = 0
    : tensor<4xf32> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @scatter_duplicate_mesh_axis(
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [0, 0]
    scatter_axis = 0 root = [0, 0]
    : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @scatter_invalid_dynamic_dimension(
    %arg0 : tensor<?xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected dynamic, but got 2.}}
  %0 = mesh.scatter %arg0 on @mesh0
    scatter_axis = 0 root = []
    : (tensor<?xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @scatter_invalid_static_dimension_size(
    %arg0 : tensor<3xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{Dimension size mismatch for result axis 0. Expected 1, but got 2.}}
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [0]
    scatter_axis = 0 root = [1]
    : (tensor<3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

mesh.mesh @mesh0(shape = 3)

func.func @scatter_invalid_operand_static_dimension_size(
    %arg0 : tensor<4xf32>) -> tensor<?xf32> {
  // expected-error@+1 {{Operand dimension size 4 is not divisible by collective device group size 3 for tensor axis 0.}}
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [0]
    scatter_axis = 0 root = [1]
    : (tensor<4xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @scatter_root_dimension_out_of_bounds(
    %arg0 : tensor<3xi8>) -> tensor<1xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "root". Got 3, but expected value in the range [0, 2].}}
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [0]
    scatter_axis = 0 root = [3]
    : (tensor<3xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @scatter_root_wrong_number_dimensions(
    %arg0 : tensor<3xi8>) -> tensor<1xi8> {
  // expected-error@+1 {{In-group device "root" has unexpected multi-index size 2. Expected 1.}}
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [0]
    scatter_axis = 0 root = [2, 2]
    : (tensor<3xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @send_destination_dimension_out_of_bounds(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{Out of bounds coordinate 0 for in-group device "destination". Got 3, but expected value in the range [0, 2].}}
  %0 = mesh.send %arg0 on @mesh0 mesh_axes = [0]
    destination = [3]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @send_destination_wrong_number_dimensions(
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // expected-error@+1 {{In-group device "destination" has unexpected multi-index size 2. Expected 1.}}
  %0 = mesh.send %arg0 on @mesh0 mesh_axes = [0]
    destination = [2, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

mesh.mesh @mesh0(shape = 3x?)

func.func @send_different_input_and_result_type(
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // expected-error@+1 {{'mesh.send' op failed to verify that all of {input, result} have same element type}}
  %0 = mesh.send %arg0 on @mesh0 mesh_axes = [0]
    destination = [2]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// -----

func.func @shift_invalid_mesh_symbol(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{Undefined required mesh symbol "this_mesh_symbol_does_not_exist".}}
  %0 = mesh.shift %arg0 on @this_mesh_symbol_does_not_exist
    shift_axis = 0 offset = -2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @shift_invalid_mesh_axis(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{0-based mesh axis index 2 is out of bounds. The referenced mesh "mesh0" is of rank 2.}}
  %0 = mesh.shift %arg0 on @mesh0 mesh_axes = [2]
        shift_axis = 2 offset = -2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @shift_duplicate_mesh_axis(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{Mesh axes contains duplicate elements.}}
  %0 = mesh.shift %arg0 on @mesh0 mesh_axes = [0, 1, 0]
    shift_axis = 0 offset = -2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @shift_invalid_tensor_dimension_size(
    %arg0 : tensor<4xi8>) -> tensor<5xi8> {
  // expected-error@+1 {{'mesh.shift' op requires the same shape for all operands and results}}
  %0 = mesh.shift %arg0 on @mesh0 mesh_axes = [0]
    shift_axis = 0 offset = 2
    : tensor<4xi8> -> tensor<5xi8>
  return %0 : tensor<5xi8>
}

// -----

mesh.mesh @mesh0(shape = 2x4)

func.func @shift_invalid_shift_axis(
    %arg0 : tensor<4xi8>) -> tensor<4xi8> {
  // expected-error@+1 {{Invalid shift axis 1. It must be one of the grouping mesh axes.}}
  %0 = mesh.shift %arg0 on @mesh0 mesh_axes = [0]
    shift_axis = 1 offset = 2
    : tensor<4xi8> -> tensor<4xi8>
  return %0 : tensor<4xi8>
}
