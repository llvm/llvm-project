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
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[]]>>
}

// CHECK-LABEL: func @mesh_shard_encoding_1st_dim
func.func @mesh_shard_encoding_1st_dim(
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0]]>>
}

// CHECK-LABEL: func @mesh_shard_encoding_2nd_dim
func.func @mesh_shard_encoding_2nd_dim(
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh1, [[], [0]]>>) -> 
    tensor<4x8xf32, #mesh.shard<@mesh1, [[], [0]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh1, [[], [0]]>>
}

// CHECK-LABEL: func @mesh_shard_encoding_1st_and_3rd_dim
func.func @mesh_shard_encoding_1st_and_3rd_dim(
    %arg0 : tensor<4x8x16xf32, #mesh.shard<@mesh3, [[0], [], [1]]>>) -> 
            tensor<4x8x16xf32, #mesh.shard<@mesh3, [[0], [], [1]]>> {
  return %arg0 : tensor<4x8x16xf32, #mesh.shard<@mesh3, [[0], [], [1]]>>
}

// CHECK-LABEL: func @mesh_shard_op_1st_dim
func.func @mesh_shard_op_1st_dim(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_2nd_dim
func.func @mesh_shard_op_2nd_dim(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mesh.shard %arg0 to <@mesh1, [[], [0]]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @mesh_shard_op_1st_and_3rd_dim
func.func @mesh_shard_op_1st_and_3rd_dim(
    %arg0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = mesh.shard %arg0 to <@mesh3, [[0], [], [1]]> : tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @mesh_shard_op_two_users
func.func @mesh_shard_op_two_users(%arg0 : tensor<4x8xf32>) -> 
                                  (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = mesh.shard %arg0 to <@mesh0, [[0]]> : tensor<4x8xf32>
  %1 = mesh.shard %0 to <@mesh0, [[1]]> {as_result = false} : tensor<4x8xf32>
  %2 = mesh.shard %0 to <@mesh0, [[2]]> {as_result = false} : tensor<4x8xf32>
  return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}
