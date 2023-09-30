// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1 {{rank of cluster is expected to be a positive integer}}
mesh.cluster @mesh0(rank = 0)

// -----

// expected-error@+1 {{rank of dim_sizes is not expected to be larger than rank of cluster}}
mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 3, 4])

// -----

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

func.func @mesh_shard_op_stacked_true_true(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mesh.shard %arg0 to <@mesh0, [[], [0]]> : tensor<4x8xf32>
  // expected-error@+1 {{two mesh.shard ops with as_result = true are not expected to be stacked together}}
  %1 = mesh.shard %0 to <@mesh0, [[], [0, 1]]> : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

func.func @mesh_shard_op_stacked_false_false(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mesh.shard %arg0 to <@mesh0, [[], [0]]> {as_result = false} : tensor<4x8xf32>
  // expected-error@+1 {{two mesh.shard ops with as_result = false are not expected to be stacked together}}
  %1 = mesh.shard %0 to <@mesh0, [[], [0, 1]]> {as_result = false} : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

mesh.cluster @mesh0(rank = 2, dim_sizes = [2, 4])

func.func @mesh_shard_ops_on_same_tensor(%arg0 : tensor<4x8xf32>) -> 
                                        (tensor<4x8xf32>, tensor<4x8xf32>) {
  // expected-error@+1 {{when than one mesh.shard ops operate on the same tensor, all of their as_result attributes are expected to be false}}
  %0 = mesh.shard %arg0 to <@mesh0, [[], [0]]> : tensor<4x8xf32>
  %1 = mesh.shard %arg0 to <@mesh0, [[], [0, 1]]> : tensor<4x8xf32>
  return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

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
