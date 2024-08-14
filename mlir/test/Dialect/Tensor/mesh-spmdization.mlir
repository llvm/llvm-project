// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(mesh-spmdization,test-constant-fold))" \
// RUN:   %s | FileCheck %s

mesh.mesh @mesh_1d_4(shape = 4)

// CHECK-LABEL: func @tensor_empty_static_sharded_dims_sizes
func.func @tensor_empty_static_sharded_dims_sizes() -> () {
  %b = tensor.empty() : tensor<8x16xf32>
  %sharding = mesh.sharding @mesh_1d_4 split_axes = [[0]] sharded_dims_sizes = [1, 3, 3, 1] : !mesh.sharding
  %sharded= mesh.shard %b to %sharding : tensor<8x16xf32>
  // CHECK:  %[[sharding:.*]] = mesh.sharding @mesh_1d_4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [1, 3, 3, 1] : !mesh.sharding
  // CHECK:  %[[proc_linear_idx:.*]] = mesh.process_linear_index on @mesh_1d_4 : index
  // CHECK:  %[[V0:.*]]:2 = mesh.shard_shape 8x16 %[[sharding]] %[[proc_linear_idx]] : index, index
  // CHECK:  tensor.empty(%[[V0]]#0) : tensor<?x16xf32>

  return
}

// CHECK-LABEL: func @tensor_empty_dynamic_sharded_dims_sizes
// CHECK-SAME: %[[A0:.*]]: index
func.func @tensor_empty_dynamic_sharded_dims_sizes(%arg0 : index) -> () {
  %b = tensor.empty(%arg0) : tensor<8x?xf32>
  %sharding = mesh.sharding @mesh_1d_4 split_axes = [[0]] sharded_dims_sizes = [1, 3, 3, 1] : !mesh.sharding
  %sharded= mesh.shard %b to %sharding : tensor<8x?xf32>
  // CHECK:  %[[sharding:.*]] = mesh.sharding @mesh_1d_4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [1, 3, 3, 1] : !mesh.sharding
  // CHECK:  %[[proc_linear_idx:.*]] = mesh.process_linear_index on @mesh_1d_4 : index
  // CHECK:  %[[V0:.*]]:2 = mesh.shard_shape 8x? %[[sharding]] %[[proc_linear_idx]] : index, index
  // CHECK:  tensor.empty(%[[V0]]#0, %[[A0]]) : tensor<?x?xf32>

  return
}

// CHECK-LABEL: func @tensor_empty_same_static_dims_sizes
func.func @tensor_empty_same_static_dims_sizes() -> () {
  %b = tensor.empty() : tensor<8x16xf32>
  %sharding = mesh.sharding @mesh_1d_4 split_axes = [[0]] sharded_dims_sizes = [4, 4, 4, 4] : !mesh.sharding
  %sharded= mesh.shard %b to %sharding : tensor<8x16xf32>
  // CHECK-NEXT:  tensor.empty() : tensor<4x16xf32>

  return
}
