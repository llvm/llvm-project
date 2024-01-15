// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: mesh.cluster @mesh0
mesh.cluster @mesh0(shape = 2x2x4)

// CHECK: mesh.cluster @mesh1(shape = 4x?)
mesh.cluster @mesh1(shape = 4x?)

// CHECK: mesh.cluster @mesh2(shape = ?x4)
mesh.cluster @mesh2(shape = ?x4)

// CHECK: mesh.cluster @mesh3(shape = ?x?)
mesh.cluster @mesh3(shape = ?x?)

mesh.cluster @mesh4(shape = 3)

// CHECK: mesh.cluster @mesh5(shape = ?)
mesh.cluster @mesh5(shape = ?)

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

// CHECK-LABEL: func @cluster_shape
func.func @cluster_shape() -> (index, index) {
  // CHECK: %[[RES:.*]]:2 = mesh.cluster_shape @mesh0 axes = [0, 1] : index, index
  %0:2 = mesh.cluster_shape @mesh0 axes = [0, 1] : index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func @cluster_shape_default_axes
func.func @cluster_shape_default_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = mesh.cluster_shape @mesh0 : index, index, index
  %0:3 = mesh.cluster_shape @mesh0 : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @cluster_shape_empty_axes
func.func @cluster_shape_empty_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = mesh.cluster_shape @mesh0 : index, index, index
  %0:3 = mesh.cluster_shape @mesh0 axes = [] : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_multi_index
func.func @process_multi_index() -> (index, index) {
  // CHECK: %[[RES:.*]]:2 = mesh.process_multi_index on @mesh0 axes = [0, 1] : index, index
  %0:2 = mesh.process_multi_index on @mesh0 axes = [0, 1] : index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func @process_multi_index_default_axes
func.func @process_multi_index_default_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = mesh.process_multi_index on @mesh0 : index, index, index
  %0:3 = mesh.process_multi_index on @mesh0 : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_multi_index_empty_axes
func.func @process_multi_index_empty_axes() -> (index, index, index) {
  // CHECK: %[[RES:.*]]:3 = mesh.process_multi_index on @mesh0 : index, index, index
  %0:3 = mesh.process_multi_index on @mesh0 axes = [] : index, index, index
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_linear_index
func.func @process_linear_index() -> index {
  // CHECK: %[[RES:.*]] = mesh.process_linear_index on @mesh0 : index
  %0 = mesh.process_linear_index on @mesh0 : index
  // CHECK: return %[[RES]] : index
  return %0 : index
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

// CHECK-LABEL: func @broadcast_static_root
func.func @broadcast_static_root(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<3x6xi8> {
  // CHECK-NEXT: mesh.broadcast %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<3x6xi8>) -> tensor<3x6xi8>
  %0 = mesh.broadcast %arg0 on @mesh0 mesh_axes = [0, 2]
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
  // CHECK-NEXT: mesh.broadcast %[[ARG0]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<3x6xi8>, index) -> tensor<3x6xi8>
  %0 = mesh.broadcast %arg0 on @mesh0 mesh_axes = [0, 2]
    root = [1, %arg1]
    : (tensor<3x6xi8>, index) -> tensor<3x6xi8>
  return %0 : tensor<3x6xi8>
}

// CHECK-LABEL: func @gather_static_root
func.func @gather_static_root(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x6xi8>
    %arg0 : tensor<3x6xi8>) -> tensor<24x6xi8> {
  // CHECK-NEXT: mesh.gather %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: gather_axis = 0
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<3x6xi8>) -> tensor<24x6xi8>
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0, 2]
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
  // CHECK-NEXT: mesh.gather %[[ARG0]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: gather_axis = 0
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<3x6xi8>, index) -> tensor<24x6xi8>
  %0 = mesh.gather %arg0 on @mesh0 mesh_axes = [0, 2]
    gather_axis = 0
    root = [1, %arg1]
    : (tensor<3x6xi8>, index) -> tensor<24x6xi8>
  return %0 : tensor<24x6xi8>
}

// CHECK-LABEL: func @receive_static_source
func.func @receive_static_source(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: mesh.recv %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: source = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi8>
  %0 = mesh.recv %arg0 on @mesh0 mesh_axes = [0, 2]
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
  // CHECK-NEXT: mesh.recv %[[ARG0]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: source = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<2xi8>, index) -> tensor<2xi8>
  %0 = mesh.recv %arg0 on @mesh0 mesh_axes = [0, 2]
    source = [1, %arg1]
    : (tensor<2xi8>, index) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @receive_no_source
func.func @receive_no_source(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: mesh.recv %[[ARG]]
  // CHECK-NOT: source
  %0 = mesh.recv %arg0 on @mesh0 mesh_axes = [0, 2]
    : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @reduce_static_root
func.func @reduce_static_root(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: mesh.reduce %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi8>
  %0 = mesh.reduce %arg0 on @mesh0 mesh_axes = [0, 2]
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
  // CHECK-NEXT: mesh.reduce %[[ARG0]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<2xi8>, index) -> tensor<2xi8>
  %0 = mesh.reduce %arg0 on @mesh0 mesh_axes = [0, 2]
    root = [1, %arg1]
    : (tensor<2xi8>, index) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @reduce_different_return_element_type
func.func @reduce_different_return_element_type(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi16> {
  // CHECK-NEXT: mesh.reduce %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: root = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi16>
  %0 = mesh.reduce %arg0 on @mesh0 mesh_axes = [0, 2]
    root = [0, 1]
    : (tensor<2xi8>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
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

// CHECK-LABEL: func @scatter_static_dimensions
func.func @scatter_static_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x1xf32> {
  // CHECK-NEXT: mesh.scatter %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [2]
  // CHECK-SAME: scatter_axis = 1 root = [1]
  // CHECK-SAME: : (tensor<3x4xf32>) -> tensor<3x1xf32>
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [2]
    scatter_axis = 1 root = [1]
    : (tensor<3x4xf32>) -> tensor<3x1xf32>
  return %0 : tensor<3x1xf32>
}

// CHECK-LABEL: func @scatter_dynamic_dimensions
func.func @scatter_dynamic_dimensions(
    // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
    %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: mesh.scatter %[[ARG]]
  // CHECK-SAME: on @mesh3 mesh_axes = [0, 1]
  // CHECK-SAME: scatter_axis = 0 root = [1, 2]
  // CHECK-SAME: : (tensor<?xf32>) -> tensor<?xf32>
  %0 = mesh.scatter %arg0 on @mesh3 mesh_axes = [0, 1]
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
  // CHECK-NEXT: mesh.scatter %[[ARG0]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: scatter_axis = 0
  // CHECK-SAME: root = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<8xi8>, index) -> tensor<1xi8>
  %0 = mesh.scatter %arg0 on @mesh0 mesh_axes = [0, 2]
    scatter_axis = 0
    root = [1, %arg1]
    : (tensor<8xi8>, index) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: func @send_static_destination
func.func @send_static_destination(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: mesh.send %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: destination = [0, 1]
  // CHECK-SAME: : (tensor<2xi8>) -> tensor<2xi8>
  %0 = mesh.send %arg0 on @mesh0 mesh_axes = [0, 2]
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
  // CHECK-NEXT: mesh.send %[[ARG0]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: destination = [1, %[[ARG1]]]
  // CHECK-SAME: : (tensor<2xi8>, index) -> tensor<2xi8>
  %0 = mesh.send %arg0 on @mesh0 mesh_axes = [0, 2]
    destination = [1, %arg1]
    : (tensor<2xi8>, index) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: func @shift
func.func @shift(
    // CHECK-SAME: %[[ARG:.*]]: tensor<2xi8>
    %arg0 : tensor<2xi8>) -> tensor<2xi8> {
  // CHECK-NEXT: mesh.shift %[[ARG]]
  // CHECK-SAME: on @mesh0 mesh_axes = [0, 2]
  // CHECK-SAME: shift_axis = 2 offset = -2 rotate
  // CHECK-SAME: : tensor<2xi8> -> tensor<2xi8>
  %0 = mesh.shift %arg0 on @mesh0 mesh_axes = [0, 2]
    shift_axis = 2 offset = -2 rotate
    : tensor<2xi8> -> tensor<2xi8>
  return %0 : tensor<2xi8>
}
