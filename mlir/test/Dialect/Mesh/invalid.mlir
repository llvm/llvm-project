// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{rank of cluster is expected to be a positive integer}}
mesh.cluster @mesh0(rank = 0)

// -----

// expected-error@+1 {{rank of dim_sizes is not expected to be larger than rank of cluster}}
mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 3, 4])

// -----

// expected-error@+1 {{dimension size of a mesh cluster is expected to be non-negative}}
mesh.cluster @mesh0(rank = 2, dim_sizes = [-1])

// -----

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

func.func @mesh_axis_duplicated(
    // expected-error@+1 {{mesh axis duplicated}}
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0], [0]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[0], [0]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0], [0]]>>
}

// -----

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

func.func @mesh_axis_duplicated_2(
    // expected-error@+1 {{mesh axis duplicated}}
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0, 0]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[0, 0]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[0, 0]]>>
}

// -----

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

func.func @mesh_axis_negtive(
    // expected-error@+1 {{mesh axis is expected to be non-negative}}
    %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[-1]]>>) -> 
            tensor<4x8xf32, #mesh.shard<@mesh0, [[-1]]>> {
  return %arg0 : tensor<4x8xf32, #mesh.shard<@mesh0, [[-1]]>>
}
