// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: shard.grid @grid0
shard.grid @grid0(shape = 2x2x4)

// CHECK: shard.grid @grid1(shape = 4x?)
shard.grid @grid1(shape = 4x?)

// CHECK: shard.grid @grid2(shape = ?x4)
shard.grid @grid2(shape = ?x4)

// CHECK: shard.grid @grid3(shape = ?x?)
shard.grid @grid3(shape = ?x?)

shard.grid @grid4(shape = 3)

// CHECK: shard.grid @grid5(shape = ?)
shard.grid @grid5(shape = ?)

// CHECK-LABEL: func @grid_shard_op_fully_replicated
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @grid_shard_op_fully_replicated(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[S:.*]] = shard.sharding @grid0 split_axes = {{\[\[}}]] : !shard.sharding
  %s = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
  // CHECK-NEXT: shard.shard %[[ARG]] to %[[S]] : tensor<4x8xf32>
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @grid_shard_op_1st_dim
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @grid_shard_op_1st_dim(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[S:.*]] = shard.sharding @grid0 split_axes = {{\[\[}}0]] : !shard.sharding
  %s = shard.sharding @grid0 split_axes = [[0]] : !shard.sharding

  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @grid_shard_op_2nd_dim
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @grid_shard_op_2nd_dim(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[S:.*]] = shard.sharding @grid1 split_axes = {{\[\[}}], [0]] : !shard.sharding
  %s = shard.sharding @grid1 split_axes = [[], [0]] : !shard.sharding
  // CHECK-NEXT: shard.shard %[[ARG]] to %[[S]] : tensor<4x8xf32>
  %0 = shard.shard %arg0 to %s : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @grid_shard_op_1st_and_3rd_dim
func.func @grid_shard_op_1st_and_3rd_dim(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8x16xf32>
    %arg0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  // CHECK-NEXT: %[[S:.*]] = shard.sharding @grid3 split_axes = {{\[\[}}0], [], [1]] : !shard.sharding
  %s = shard.sharding @grid3 split_axes = [[0], [], [1]] : !shard.sharding
  // CHECK-NEXT: shard.shard %[[ARG]] to %[[S]] : tensor<4x8x16xf32>
  %0 = shard.shard %arg0 to %s : tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @grid_shard_op_two_users
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @grid_shard_op_two_users(%arg0 : tensor<4x8xf32>) ->
                                  (tensor<4x8xf32>, tensor<4x8xf32>) {
  // CHECK-NEXT: %[[V0:.*]] = shard.sharding @grid0 split_axes = {{\[\[}}0]] : !shard.sharding
  %s0 = shard.sharding @grid0 split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<4x8xf32>
  // CHECK-DAG: shard.sharding @grid0 split_axes = {{\[\[}}1]] : !shard.sharding
  %s1 = shard.sharding @grid0 split_axes = [[1]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<4x8xf32>
  // CHECK-DAG: shard.sharding @grid0 split_axes = {{\[\[}}2]] : !shard.sharding
  %s2 = shard.sharding @grid0 split_axes = [[2]] : !shard.sharding
  %2 = shard.shard %0 to %s2 annotate_for_users : tensor<4x8xf32>
  return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @grid_shard_halo_sizes
func.func @grid_shard_halo_sizes() -> () {
  // CHECK: %[[C3:.*]] = arith.constant 3 : i64
  %c3 = arith.constant 3 : i64
  // CHECK: shard.sharding @grid4 split_axes = {{\[\[}}0]] halo_sizes = [1, 4] : !shard.sharding
  %sharding1 = shard.sharding @grid4 split_axes = [[0]] halo_sizes = [1, 4] : !shard.sharding
  // CHECK: shard.sharding @grid4 split_axes = {{\[\[}}0]] halo_sizes = [4, %[[C3]]] : !shard.sharding
  %sharding2 = shard.sharding @grid4 split_axes = [[0]] halo_sizes = [4, %c3] : !shard.sharding
  return
}

// CHECK-LABEL: func @grid_shard_dims_sizes
func.func @grid_shard_dims_sizes() -> () {
  // CHECK: %[[C3:.*]] = arith.constant 3 : i64
  %c3 = arith.constant 3 : i64
  // CHECK: shard.sharding @grid4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 1, 4, 6] : !shard.sharding
  %sharding1 = shard.sharding @grid4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !shard.sharding
  // CHECK: shard.sharding @grid4 split_axes = {{\[\[}}0]] sharded_dims_offsets = [0, 2, %[[C3]], 5] : !shard.sharding
  %sharding2 = shard.sharding @grid4 split_axes = [[0]] sharded_dims_offsets = [0, 2, %c3, 5] : !shard.sharding
  return
}

// CHECK-LABEL: func @grid_shard_shape
func.func @grid_shard_shape() {
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  // CHECK-NEXT: %[[S:.*]] = shard.sharding @grid0 split_axes = {{\[\[}}]] : !shard.sharding
  %s = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
  // CHECK-NEXT: shard.shard_shape dims = [8, %[[C3]]
  // CHECK-SAME: ] sharding = %[[S]] device = [%[[C3]]
  // CHECK-SAME: ] : index, index
  %shp:2 = shard.shard_shape dims = [8, %c3] sharding = %s device = [%c3] : index, index
  // CHECK-NEXT: shard.shard_shape dims = [8, 4] sharding = %[[S]] device = [3] : index, index
  %shp1:2 = shard.shard_shape dims = [8, 4] sharding = %s device = [3] : index, index
  return
}

// CHECK-LABEL: func @grid_get_sharding
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @grid_get_sharding(%arg0 : tensor<4x8xf32>) -> !shard.sharding {
  // CHECK-NEXT: shard.get_sharding %[[ARG]] : tensor<4x8xf32> -> !shard.sharding
  %0 = shard.get_sharding %arg0 : tensor<4x8xf32> -> !shard.sharding
  return %0 : !shard.sharding
}

// CHECK-LABEL: func @grid_shape
func.func @grid_shape() -> (index, index) {
  // CHECK: %[[RES:.*]]:2 = shard.grid_shape @grid0 axes = [0, 1] : index, index
  %0:2 = shard.grid_shape @grid0 axes = [0, 1] : index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func @grid_shape_default_axes
func.func @grid_shape_default_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = shard.grid_shape @grid0 : index, index, index
  %0:3 = shard.grid_shape @grid0 : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @grid_shape_empty_axes
func.func @grid_shape_empty_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = shard.grid_shape @grid0 : index, index, index
  %0:3 = shard.grid_shape @grid0 axes = [] : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_multi_index
func.func @process_multi_index() -> (index, index) {
  // CHECK: %[[RES:.*]]:2 = shard.process_multi_index on @grid0 axes = [0, 1] : index, index
  %0:2 = shard.process_multi_index on @grid0 axes = [0, 1] : index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func @process_multi_index_default_axes
func.func @process_multi_index_default_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = shard.process_multi_index on @grid0 : index, index, index
  %0:3 = shard.process_multi_index on @grid0 : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_multi_index_empty_axes
func.func @process_multi_index_empty_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = shard.process_multi_index on @grid0 : index, index, index
  %0:3 = shard.process_multi_index on @grid0 axes = [] : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_linear_index
func.func @process_linear_index() -> index {
  // CHECK: %[[RES:.*]] = shard.process_linear_index on @grid0 : index
  %0 = shard.process_linear_index on @grid0 : index
  // CHECK: return %[[RES]] : index
  return %0 : index
}

// CHECK-LABEL: func @all_reduce
func.func @all_reduce(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x4xf64> {
  // CHECK-NEXT: shard.all_reduce %[[ARG]] on @grid0 grid_axes = [1, 0] reduction = max
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x4xf64>
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [1, 0] reduction = max
    : tensor<3x4xf32> -> tensor<3x4xf64>
  return %0 : tensor<3x4xf64>
}

// CHECK-LABEL: func @all_gather
func.func @all_gather(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x16xf32> {
  // CHECK-NEXT: shard.all_gather %[[ARG]] on @grid0 grid_axes = [2] gather_axis = 1
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x16xf32>
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [2] gather_axis = 1
    : tensor<3x4xf32> -> tensor<3x16xf32>
  return %0 : tensor<3x16xf32>
}

// CHECK-LABEL: func @all_gather_dynamic_dims_in_tensor
func.func @all_gather_dynamic_dims_in_tensor(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xf32>
    %arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NEXT: shard.all_gather %[[ARG]] on @grid0 grid_axes = [2] gather_axis = 1
  // CHECK-SAME: : tensor<?x?xf32> -> tensor<?x?xf32>
  %0 = shard.all_gather %arg0 on @grid0 grid_axes = [2] gather_axis = 1
    : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @all_gather_dynamic_dims_in_grid
func.func @all_gather_dynamic_dims_in_grid(
    // CHECK-SAME: %[[ARG:.*]]: tensor<5x6xf32>
    %arg0 : tensor<5x6xf32>) -> tensor<5x?xf32> {
  // CHECK-NEXT: shard.all_gather %[[ARG]] on @grid3 grid_axes = [1] gather_axis = 1
  // CHECK-SAME: : tensor<5x6xf32> -> tensor<5x?xf32>
  %0 = shard.all_gather %arg0 on @grid3 grid_axes = [1] gather_axis = 1
    : tensor<5x6xf32> -> tensor<5x?xf32>
  return %0 : tensor<5x?xf32>
}

// CHECK-LABEL: func @all_slice_static_dimensions
func.func @all_slice_static_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x1xf32> {
  // CHECK-NEXT: shard.all_slice %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [2] slice_axis = 1
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x1xf32>
  %0 = shard.all_slice %arg0 on @grid0 grid_axes = [2] slice_axis = 1
    : tensor<3x4xf32> -> tensor<3x1xf32>
  return %0 : tensor<3x1xf32>
}

// CHECK-LABEL: func @all_slice_dynamic_dimensions
func.func @all_slice_dynamic_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: shard.all_slice %[[ARG]]
  // CHECK-SAME: on @grid3 grid_axes = [0, 1] slice_axis = 0
  // CHECK-SAME: : tensor<?xf32> -> tensor<?xf32>
  %0 = shard.all_slice %arg0 on @grid3 grid_axes = [0, 1] slice_axis = 0
    : tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @all_to_all
func.func @all_to_all(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // CHECK-NEXT: shard.all_to_all %[[ARG]]
  // CHECK-SAME: on @grid4 split_axis = 1 concat_axis = 0
  // CHECK-SAME: : tensor<3x6xi8> -> tensor<3x6xi8>
  %0 = shard.all_to_all %arg0 on @grid4
    split_axis = 1 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// CHECK-LABEL: func @all_to_all_dynamic_dims_in_result
func.func @all_to_all_dynamic_dims_in_result(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<3x?xi8> {
  // CHECK-NEXT: shard.all_to_all %[[ARG]]
  // CHECK-SAME: on @grid4 split_axis = 1 concat_axis = 0
  // CHECK-SAME: : tensor<3x6xi8> -> tensor<3x?xi8>
  %0 = shard.all_to_all %arg0 on @grid4
    split_axis = 1 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x?xi8>
  return %0 : tensor<3x?xi8>
}

// CHECK-LABEL: func @all_to_all_same_split_concat_dim_with_dynamic_device_group_size
func.func @all_to_all_same_split_concat_dim_with_dynamic_device_group_size(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3xi8>
    %arg0 : tensor<3xi8>) -> tensor<3xi8> {
  // CHECK-NEXT: shard.all_to_all %[[ARG]]
  // CHECK-SAME: @grid4 split_axis = 0 concat_axis = 0
  // CHECK-SAME: : tensor<3xi8> -> tensor<3xi8>
  %0 = shard.all_to_all %arg0 on @grid4
    split_axis = 0 concat_axis = 0
    : tensor<3xi8> -> tensor<3xi8>
  return %0 : tensor<3xi8>
}

// CHECK-LABEL: func @all_to_all_non_divisible_split_axis_size
func.func @all_to_all_non_divisible_split_axis_size(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2x3xi8>
    %arg0 : tensor<2x3xi8>) -> tensor<?x12xi8> {
  // CHECK-NEXT: shard.all_to_all %[[ARG]]
  // CHECK-SAME: @grid0 grid_axes = [0, 1] split_axis = 0 concat_axis = 1
  // CHECK-SAME: : tensor<2x3xi8> -> tensor<?x12xi8>
  %0 = shard.all_to_all %arg0 on @grid0 grid_axes = [0, 1]
    split_axis = 0 concat_axis = 1
    : tensor<2x3xi8> -> tensor<?x12xi8>
  return %0 : tensor<?x12xi8>
}

// CHECK-LABEL: func @broadcast_static_root
func.func @broadcast_static_root(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // CHECK-NEXT: shard.broadcast %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<3x6xi8>) -> tensor<3x6xi8>
  %0 = shard.broadcast %arg0 on @grid0 grid_axes = [0, 2]
    root = [0, 1]
    : (tensor<3x6xi8>) -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// CHECK-LABEL: func @broadcast_dynamic_root
func.func @broadcast_dynamic_root(
    // CHECK-SAME: %[[ARG0:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>,
    // CHECK-SAME: %[[ARG1:.*]]: index
    %arg1 : index
    ) -> tensor<3x6xi8> {
  // CHECK-NEXT: shard.broadcast %[[ARG0]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<3x6xi8>, index) -> tensor<3x6xi8>
  %0 = shard.broadcast %arg0 on @grid0 grid_axes = [0, 2]
    root = [1, %arg1]
    : (tensor<3x6xi8>, index) -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// CHECK-LABEL: func @gather_static_root
func.func @gather_static_root(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<24x6xi8> {
  // CHECK-NEXT: shard.gather %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: gather_axis = 0
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<3x6xi8>) -> tensor<24x6xi8>
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0, 2]
    gather_axis = 0
    root = [0, 1]
    : (tensor<3x6xi8>) -> tensor<24x6xi8>
  return %0 : tensor<24x6xi8>
}

// CHECK-LABEL: func @gather_dynamic_root
func.func @gather_dynamic_root(
    // CHECK-SAME: %[[ARG0:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>,
    // CHECK-SAME: %[[ARG1:.*]]: index
    %arg1 : index
    ) -> tensor<24x6xi8> {
  // CHECK-NEXT: shard.gather %[[ARG0]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: gather_axis = 0
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<3x6xi8>, index) -> tensor<24x6xi8>
  %0 = shard.gather %arg0 on @grid0 grid_axes = [0, 2]
    gather_axis = 0
    root = [1, %arg1]
    : (tensor<3x6xi8>, index) -> tensor<24x6xi8>
  return %0 : tensor<24x6xi8>
}

// CHECK-LABEL: func @receive_static_source
func.func @receive_static_source(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: shard.recv %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: source = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi8>
  %0 = shard.recv %arg0 on @grid0 grid_axes = [0, 2]
    source = [0, 1]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @receive_dynamic_source
func.func @receive_dynamic_source(
    // CHECK-SAME: %[[ARG0:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>,
    // CHECK-SAME: %[[ARG1:.*]]: index
    %arg1 : index
    ) -> tensor<2xi8> {
  // CHECK-NEXT: shard.recv %[[ARG0]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: source = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<2xi8>, index) -> tensor<2xi8>
  %0 = shard.recv %arg0 on @grid0 grid_axes = [0, 2]
    source = [1, %arg1]
    : (tensor<2xi8>, index) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @receive_no_source
func.func @receive_no_source(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: shard.recv %[[ARG]]
  // CHECK-NOT: source
  %0 = shard.recv %arg0 on @grid0 grid_axes = [0, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @reduce_static_root
func.func @reduce_static_root(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: shard.reduce %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi8>
  %0 = shard.reduce %arg0 on @grid0 grid_axes = [0, 2]
    root = [0, 1]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @reduce_dynamic_root
func.func @reduce_dynamic_root(
    // CHECK-SAME: %[[ARG0:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>,
    // CHECK-SAME: %[[ARG1:.*]]: index
    %arg1 : index
    ) -> tensor<2xi8> {
  // CHECK-NEXT: shard.reduce %[[ARG0]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<2xi8>, index) -> tensor<2xi8>
  %0 = shard.reduce %arg0 on @grid0 grid_axes = [0, 2]
    root = [1, %arg1]
    : (tensor<2xi8>, index) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @reduce_different_return_element_type
func.func @reduce_different_return_element_type(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // CHECK-NEXT: shard.reduce %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi16>
  %0 = shard.reduce %arg0 on @grid0 grid_axes = [0, 2]
    root = [0, 1]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}

// CHECK-LABEL: func @reduce_scatter_static_dimensions
func.func @reduce_scatter_static_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x1xf64> {
  // CHECK-NEXT: shard.reduce_scatter %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [2] reduction = max scatter_axis = 1
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x1xf64>
  %0 = shard.reduce_scatter %arg0 on @grid0 grid_axes = [2]
    reduction = max scatter_axis = 1
    : tensor<3x4xf32> -> tensor<3x1xf64>
  return %0 : tensor<3x1xf64>
}

// CHECK-LABEL: func @reduce_scatter_dynamic_dimensions
func.func @reduce_scatter_dynamic_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
    %arg0 : tensor<?xf32>) -> tensor<?xf64> {
  // CHECK-NEXT: shard.reduce_scatter %[[ARG]]
  // CHECK-SAME: on @grid3 grid_axes = [0, 1] scatter_axis = 0
  // CHECK-SAME: : tensor<?xf32> -> tensor<?xf64>
  %0 = shard.reduce_scatter %arg0 on @grid3 grid_axes = [0, 1] scatter_axis = 0
    : tensor<?xf32> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}

// CHECK-LABEL: func @scatter_static_dimensions
func.func @scatter_static_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x1xf32> {
  // CHECK-NEXT: shard.scatter %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [2]
  // CHECK-SAME: scatter_axis = 1 root = [1]
  // CHECK-SAME: : (tensor<3x4xf32>) -> tensor<3x1xf32>
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [2]
    scatter_axis = 1 root = [1]
    : (tensor<3x4xf32>) -> tensor<3x1xf32>
  return %0 : tensor<3x1xf32>
}

// CHECK-LABEL: func @scatter_dynamic_dimensions
func.func @scatter_dynamic_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: shard.scatter %[[ARG]]
  // CHECK-SAME: on @grid3 grid_axes = [0, 1]
  // CHECK-SAME: scatter_axis = 0 root = [1, 2]
  // CHECK-SAME: : (tensor<?xf32>) -> tensor<?xf32>
  %0 = shard.scatter %arg0 on @grid3 grid_axes = [0, 1]
    scatter_axis = 0 root = [1, 2]
    : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @scatter_dynamic_root
func.func @scatter_dynamic_root(
    // CHECK-SAME: %[[ARG0:.*]]: tensor<8xi8>
    %arg0 : tensor<8xi8>,
    // CHECK-SAME: %[[ARG1:.*]]: index
    %arg1 : index
    ) -> tensor<1xi8> {
  // CHECK-NEXT: shard.scatter %[[ARG0]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: scatter_axis = 0
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<8xi8>, index) -> tensor<1xi8>
  %0 = shard.scatter %arg0 on @grid0 grid_axes = [0, 2]
    scatter_axis = 0
    root = [1, %arg1]
    : (tensor<8xi8>, index) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: func @send_static_destination
func.func @send_static_destination(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: shard.send %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: destination = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi8>
  %0 = shard.send %arg0 on @grid0 grid_axes = [0, 2]
    destination = [0, 1]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @send_dynamic_destination
func.func @send_dynamic_destination(
    // CHECK-SAME: %[[ARG0:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>,
    // CHECK-SAME: %[[ARG1:.*]]: index
    %arg1 : index
    ) -> tensor<2xi8> {
  // CHECK-NEXT: shard.send %[[ARG0]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: destination = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<2xi8>, index) -> tensor<2xi8>
  %0 = shard.send %arg0 on @grid0 grid_axes = [0, 2]
    destination = [1, %arg1]
    : (tensor<2xi8>, index) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @shift
func.func @shift(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: shard.shift %[[ARG]]
  // CHECK-SAME: on @grid0 grid_axes = [0, 2]
  // CHECK-SAME: shift_axis = 2 offset = -2 rotate
  // CHECK-SAME: : tensor<2xi8> -> tensor<2xi8>
  %0 = shard.shift %arg0 on @grid0 grid_axes = [0, 2]
    shift_axis = 2 offset = -2 rotate
    : tensor<2xi8> -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @update_halo
func.func @update_halo(
    // CHECK-SAME: %[[ARG:.*]]: memref<12x12xi8>
    %arg0 : memref<12x12xi8>) {
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: %[[UH1:.*]] = shard.update_halo %[[ARG]] on @grid0
  // CHECK-SAME: split_axes = {{\[\[}}0]]
  // CHECK-SAME: halo_sizes = [2, %c2_i64] : memref<12x12xi8>
  %c2 = arith.constant 2 : i64
  %uh1 = shard.update_halo %arg0 on @grid0 split_axes = [[0]]
    halo_sizes = [2, %c2] : memref<12x12xi8>
  // CHECK-NEXT: %[[UH2:.*]] = shard.update_halo %[[UH1]] on @grid0
  // CHECK-SAME: split_axes = {{\[\[}}0], [1]]
  // CHECK-SAME: halo_sizes = [2, 2, %[[C2]], 2] : memref<12x12xi8>
  %uh2 = shard.update_halo %uh1 on @grid0 split_axes = [[0], [1]]
    halo_sizes = [2, 2, %c2, 2] : memref<12x12xi8>
  return
}
