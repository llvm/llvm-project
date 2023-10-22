// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: mesh.cluster @mesh0
mesh.cluster @mesh0(rank = 3, dim_sizes = [2, 2, 4])

// CHECK: mesh.cluster @mesh1
mesh.cluster @mesh1(rank = 2, dim_sizes = [4])

// CHECK: mesh.cluster @mesh2
mesh.cluster @mesh2(rank = 2, dim_sizes = [0, 4])

// CHECK: mesh.cluster @mesh3
mesh.cluster @mesh3(rank = 2)

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
