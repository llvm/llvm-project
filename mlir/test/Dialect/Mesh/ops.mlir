// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: mesh.cluster @mesh0
mesh.cluster @mesh0(rank = 3, dim_sizes = [2, 2, 4])

// CHECK: mesh.cluster @mesh1
mesh.cluster @mesh1(rank = 2, dim_sizes = [4])

// CHECK: mesh.cluster @mesh2
mesh.cluster @mesh2(rank = 2, dim_sizes = [0, 4])

// CHECK: mesh.cluster @mesh3
mesh.cluster @mesh3(rank = 2)

mesh.cluster @mesh4(rank = 1, dim_sizes = [3])

// CHECK-LABEL: func @mesh_shard_encoding_fully_replicated
func.func @mesh_shard_encoding_fully_replicated(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32, #mesh.shard<@mesh0, {{\[\[}}]]>>
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>>
}

// CHECK-LABEL: func @mesh_shard_encoding_1st_dim
func.func @mesh_shard_encoding_1st_dim(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32, #mesh.shard<@mesh0, {{\[\[}}0]]>>
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>>
}

// CHECK-LABEL: func @mesh_shard_encoding_2nd_dim
func.func @mesh_shard_encoding_2nd_dim(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32, #mesh.shard<@mesh1, {{\[\[}}], [0]]>>
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh1, [[], [0]]>>) -> 
    tensor<4x8xf32, #mesh.shard<@mesh1, [[], [0]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh1, [[], [0]]>>
}

// CHECK-LABEL: func @mesh_shard_encoding_1st_and_3rd_dim
func.func @mesh_shard_encoding_1st_and_3rd_dim(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8x16xf32, #mesh.shard<@mesh3, {{\[\[}}0], [], [1]]>>
    %arg0 : tensor<4x8x16xf32, #mesh.shard<@mesh3, [[0], [], [1]]>>) -> 
            tensor<4x8x16xf32, #mesh.shard<@mesh3, [[0], [], [1]]>> {
  return %arg0 : tensor<4x8x16xf32, #mesh.shard<@mesh3, [[0], [], [1]]>>
}

// CHECK-LABEL: func @mesh_shard_op_1st_dim
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @mesh_shard_op_1st_dim(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh0, {{\[\[}}0]]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_2nd_dim
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @mesh_shard_op_2nd_dim(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh1, {{\[\[}}], [0]]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh1, [[], [0]]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_1st_and_3rd_dim
func.func @mesh_shard_op_1st_and_3rd_dim(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8x16xf32>
    %arg0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh3, {{\[\[}}0], [], [1]]> : tensor<4x8x16xf32>
  %0 = mesh.shard %arg0 to <@mesh3, [[0], [], [1]]> : tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @mesh_shard_op_partial_max
func.func @mesh_shard_op_partial_max(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh3, {{\[\[}}0]], partial = max[1]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh3, [[0]], partial = max[1]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_partial_min
func.func @mesh_shard_op_partial_min(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh3, {{\[\[}}0]], partial = min[1]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh3, [[0]], partial = min[1]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_partial_generic
func.func @mesh_shard_op_partial_generic(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh3, {{\[\[}}0]], partial = generic[1]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh3, [[0]], partial = generic[1]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_partial_sum
func.func @mesh_shard_op_partial_sum(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh3, {{\[\[}}0]], partial = sum[1]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh3, [[0]], partial = sum[1]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_partial_sum_multi_axes
func.func @mesh_shard_op_partial_sum_multi_axes(
    // CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
    %arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: mesh.shard %[[ARG]] to <@mesh3, {{\[\[}}0]], partial = sum[1, 2]> : tensor<4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh3, [[0]], partial = sum[1, 2]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_two_users
// CHECK-SAME: %[[ARG:.*]]: tensor<4x8xf32>
func.func @mesh_shard_op_two_users(%arg0 : tensor<4x8xf32>) -> 
                                  (tensor<4x8xf32>, tensor<4x8xf32>) {
  // CHECK-NEXT: %[[V0:.*]] = mesh.shard %[[ARG]] to <@mesh0, {{\[\[}}0]]> : tensor<4x8xf32>                  
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  // CHECK-DAG: mesh.shard %[[V0]] to <@mesh0, {{\[\[}}1]]> annotate_for_users : tensor<4x8xf32>
  %1 = mesh.shard %0 to <@mesh0, [[1]]> annotate_for_users : tensor<4x8xf32>
  // CHECK-DAG: mesh.shard %[[V0]] to <@mesh0, {{\[\[}}2]]> annotate_for_users : tensor<4x8xf32>
  %2 = mesh.shard %0 to <@mesh0, [[2]]> annotate_for_users : tensor<4x8xf32>
  return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce
func.func @all_reduce(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x4xf64> {
  // CHECK-NEXT: mesh.all_reduce %[[ARG]] on @mesh0 mesh_axes = [1, 0] reduction = <max>
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x4xf64>
  %0 = mesh.all_reduce %arg0 on @mesh0 mesh_axes = [1, 0] reduction = <max>
    : tensor<3x4xf32> -> tensor<3x4xf64>
  return %0 : tensor<3x4xf64>
}

// CHECK-LABEL: func @all_gather
func.func @all_gather(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x16xf32> {
  // CHECK-NEXT: mesh.all_gather %[[ARG]] on @mesh0 mesh_axes = [2] gather_axis = 1
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x16xf32>
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [2] gather_axis = 1
    : tensor<3x4xf32> -> tensor<3x16xf32>
  return %0 : tensor<3x16xf32>
}

// CHECK-LABEL: func @all_gather_dynamic_dims_in_tensor
func.func @all_gather_dynamic_dims_in_tensor(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xf32>
    %arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NEXT: mesh.all_gather %[[ARG]] on @mesh0 mesh_axes = [2] gather_axis = 1
  // CHECK-SAME: : tensor<?x?xf32> -> tensor<?x?xf32>
  %0 = mesh.all_gather %arg0 on @mesh0 mesh_axes = [2] gather_axis = 1
    : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @all_gather_dynamic_dims_in_mesh
func.func @all_gather_dynamic_dims_in_mesh(
    // CHECK-SAME: %[[ARG:.*]]: tensor<5x6xf32>
    %arg0 : tensor<5x6xf32>) -> tensor<5x?xf32> {
  // CHECK-NEXT: mesh.all_gather %[[ARG]] on @mesh3 mesh_axes = [1] gather_axis = 1
  // CHECK-SAME: : tensor<5x6xf32> -> tensor<5x?xf32>
  %0 = mesh.all_gather %arg0 on @mesh3 mesh_axes = [1] gather_axis = 1
    : tensor<5x6xf32> -> tensor<5x?xf32>
  return %0 : tensor<5x?xf32>
}

// CHECK-LABEL: func @all_to_all
func.func @all_to_all(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // CHECK-NEXT: mesh.all_to_all %[[ARG]]
  // CHECK-SAME: on @mesh4 split_axis = 1 concat_axis = 0
  // CHECK-SAME: : tensor<3x6xi8> -> tensor<3x6xi8>
  %0 = mesh.all_to_all %arg0 on @mesh4
    split_axis = 1 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// CHECK-LABEL: func @all_to_all_dynamic_dims_in_result
func.func @all_to_all_dynamic_dims_in_result(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<3x?xi8> {
  // CHECK-NEXT: mesh.all_to_all %[[ARG]]
  // CHECK-SAME: on @mesh4 split_axis = 1 concat_axis = 0
  // CHECK-SAME: : tensor<3x6xi8> -> tensor<3x?xi8>
  %0 = mesh.all_to_all %arg0 on @mesh4
    split_axis = 1 concat_axis = 0
    : tensor<3x6xi8> -> tensor<3x?xi8>
  return %0 : tensor<3x?xi8>
}

// CHECK-LABEL: func @all_to_all_same_split_concat_dim_with_dynamic_device_group_size
func.func @all_to_all_same_split_concat_dim_with_dynamic_device_group_size(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3xi8>
    %arg0 : tensor<3xi8>) -> tensor<3xi8> {
  // CHECK-NEXT: mesh.all_to_all %[[ARG]]
  // CHECK-SAME: @mesh4 split_axis = 0 concat_axis = 0
  // CHECK-SAME: : tensor<3xi8> -> tensor<3xi8>
  %0 = mesh.all_to_all %arg0 on @mesh4
    split_axis = 0 concat_axis = 0
    : tensor<3xi8> -> tensor<3xi8>
  return %0 : tensor<3xi8>
}

// CHECK-LABEL: func @all_to_all_non_divisible_split_axis_size
func.func @all_to_all_non_divisible_split_axis_size(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2x3xi8>
    %arg0 : tensor<2x3xi8>) -> tensor<?x12xi8> {
  // CHECK-NEXT: mesh.all_to_all %[[ARG]]
  // CHECK-SAME: @mesh0 mesh_axes = [0, 1] split_axis = 0 concat_axis = 1
  // CHECK-SAME: : tensor<2x3xi8> -> tensor<?x12xi8>
  %0 = mesh.all_to_all %arg0 on @mesh0 mesh_axes = [0, 1]
    split_axis = 0 concat_axis = 1
    : tensor<2x3xi8> -> tensor<?x12xi8>
  return %0 : tensor<?x12xi8>
}

// CHECK-LABEL: func @reduce_scatter_static_dimensions
func.func @reduce_scatter_static_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x1xf64> {
  // CHECK-NEXT: mesh.reduce_scatter %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [2] reduction = <max> scatter_axis = 1
  // CHECK-SAME: : tensor<3x4xf32> -> tensor<3x1xf64>
  %0 = mesh.reduce_scatter %arg0 on @mesh0 mesh_axes = [2]
    reduction = <max> scatter_axis = 1
    : tensor<3x4xf32> -> tensor<3x1xf64>
  return %0 : tensor<3x1xf64>
}

// CHECK-LABEL: func @reduce_scatter_dynamic_dimensions
func.func @reduce_scatter_dynamic_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
    %arg0 : tensor<?xf32>) -> tensor<?xf64> {
  // CHECK-NEXT: mesh.reduce_scatter %[[ARG]]
  // CHECK-SAME: on @mesh3 mesh_axes = [0, 1] scatter_axis = 0
  // CHECK-SAME: : tensor<?xf32> -> tensor<?xf64>
  %0 = mesh.reduce_scatter %arg0 on @mesh3 mesh_axes = [0, 1] scatter_axis = 0
    : tensor<?xf32> -> tensor<?xf64>
  return %0 : tensor<?xf64>
}
