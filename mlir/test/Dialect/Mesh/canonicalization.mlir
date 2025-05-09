// RUN: mlir-opt --canonicalize %s | FileCheck %s

mesh.mesh @mesh0(shape = 2x4)

// CHECK-LABEL: func @all_reduce_empty_mesh_axes
func.func @all_reduce_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.all_reduce
  %0 = mesh.all_reduce %arg0 on @mesh0
    mesh_axes = []
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @all_reduce_empty_mesh_axes_different_return_type
func.func @all_reduce_empty_mesh_axes_different_return_type(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
// CHECK: mesh.all_reduce
  %0 = mesh.all_reduce %arg0 on @mesh0
// CHECK-NOT: mesh_axes
    mesh_axes = []
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_reduce_default_reduction
func.func @all_reduce_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  %0 = mesh.all_reduce %arg0 on @mesh0
    mesh_axes = [0]
// CHECK-NOT: reduction
    reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_to_all_empty_mesh_axes
func.func @all_to_all_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32>
    %arg0 : tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NOT: mesh.all_to_all
  %0 = mesh.all_to_all %arg0 on @mesh0
    mesh_axes = []
    split_axis = 0
    concat_axis = 0
    : tensor<8xf32> -> tensor<8xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @all_gather_empty_mesh_axes
func.func @all_gather_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.all_gather
  %0 = mesh.all_gather %arg0 on @mesh0
    mesh_axes = []
    gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @all_slice_empty_mesh_axes
func.func @all_slice_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.scatter
  %0 = mesh.all_slice %arg0 on @mesh0
    mesh_axes = []
    slice_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @broadcast_empty_mesh_axes
func.func @broadcast_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.broadcast
  %0 = mesh.broadcast %arg0 on @mesh0
    mesh_axes = []
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @gather_empty_mesh_axes
func.func @gather_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.gather
  %0 = mesh.gather %arg0 on @mesh0
    mesh_axes = []
    gather_axis = 0
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @receive_empty_mesh_axes
func.func @receive_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.recv
  %0 = mesh.recv %arg0 on @mesh0
    mesh_axes = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_empty_mesh_axes
func.func @reduce_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.reduce
  %0 = mesh.reduce %arg0 on @mesh0
    mesh_axes = []
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_mesh_axes
func.func @reduce_scatter_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.reduce_scatter
  %0 = mesh.reduce_scatter %arg0 on @mesh0
    mesh_axes = []
    scatter_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_mesh_axes_different_return_type
func.func @reduce_scatter_empty_mesh_axes_different_return_type(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
// CHECK: mesh.reduce_scatter
  %0 = mesh.reduce_scatter %arg0 on @mesh0
// CHECK-NOT: mesh_axes
    mesh_axes = []
    scatter_axis = 0
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @reduce_scatter_default_reduction
func.func @reduce_scatter_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<2xf64> {
  %0 = mesh.reduce_scatter %arg0 on @mesh0
    mesh_axes = [0]
// CHECK-NOT: reduction
    reduction = sum
    scatter_axis = 0
    : tensor<4xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func @scatter_empty_mesh_axes
func.func @scatter_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.scatter
  %0 = mesh.scatter %arg0 on @mesh0
    mesh_axes = []
    scatter_axis = 0
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @send_empty_mesh_axes
func.func @send_empty_mesh_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: mesh.send
  %0 = mesh.send %arg0 on @mesh0
    mesh_axes = []
    destination = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

mesh.mesh @mesh4x4(shape = 4x4)
// CHECK-LABEL: func @test_halo_sizes
func.func @test_halo_sizes() -> !mesh.sharding {
  %c2_i64 = arith.constant 2 : i64
  // CHECK mesh.sharding @mesh4x4 split_axes = [[0], [1]] halo_sizes = [1, 2, 2, 22] : !mesh.sharding
  %sharding = mesh.sharding @mesh4x4 split_axes = [[0], [1]] halo_sizes = [1, %c2_i64, %c2_i64, 22] : !mesh.sharding
  return %sharding : !mesh.sharding
}

// CHECK-LABEL: func @test_shard_offs
func.func @test_shard_offs() -> !mesh.sharding {
  %c2_i64 = arith.constant 2 : i64
  // CHECK mesh.sharding @mesh4x4 split_axes = [[0], [1]] sharded_dims_offsets = [0, 1, 2, 3, 4, 0, 2, 3, 4, 22] : !mesh.sharding
  %sharding = mesh.sharding @mesh4x4 split_axes = [[0], [1]] sharded_dims_offsets = [0, 1, %c2_i64, 3, 4, 0, %c2_i64, 3, 4, 22] : !mesh.sharding
  return %sharding : !mesh.sharding
}

// CHECK-LABEL: func @test_duplicate_shardops
func.func @test_duplicate_shardops() -> (tensor<1024x1024xf32>, tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0, 1]] : !mesh.sharding
  %sharding_1 = mesh.sharding @mesh4x4 split_axes = [[0, 1]] : !mesh.sharding
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_2 = mesh.sharding @mesh4x4 split_axes = [[0, 1]] : !mesh.sharding
  %sharding_annotated_2 = mesh.shard %cst_2 to %sharding_2 : tensor<1024x1024xf32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_3 = mesh.sharding @mesh4x4 split_axes = [[0, 1]] : !mesh.sharding
  %sharding_annotated_3 = mesh.shard %cst_3 to %sharding_3 : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding_annotated:%.*]] = mesh.shard [[vcst]] to [[vsharding]] : tensor<1024x1024xf32>
  %sharding_annotated_1 = mesh.shard %cst_1 to %sharding_1 : tensor<1024x1024xf32>
  // CHECK-NEXT: return [[vsharding_annotated]], [[vsharding_annotated]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>
  return %sharding_annotated_1, %sharding_annotated_2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
}

// CHECK-LABEL: func @test_duplicate_shardops_diff
func.func @test_duplicate_shardops_diff() -> (tensor<1024x1024xf32>, tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
  %sharding_1 = mesh.sharding @mesh4x4 split_axes = [[0]] : !mesh.sharding
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding_0:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0, 1]] : !mesh.sharding
  %sharding_2 = mesh.sharding @mesh4x4 split_axes = [[0, 1]] : !mesh.sharding
  // CHECK-NEXT: [[vsharding_annotated:%.*]] = mesh.shard [[vcst]] to [[vsharding_0]] : tensor<1024x1024xf32>
  %sharding_annotated_2 = mesh.shard %cst_2 to %sharding_2 : tensor<1024x1024xf32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_3 = mesh.sharding @mesh4x4 split_axes = [[0]] : !mesh.sharding
  %sharding_annotated_3 = mesh.shard %cst_3 to %sharding_3 : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding_annotated_1:%.*]] = mesh.shard [[vsharding_annotated]] to [[vsharding]] : tensor<1024x1024xf32>
  %sharding_annotated_1 = mesh.shard %cst_1 to %sharding_1 : tensor<1024x1024xf32>
  // CHECK-NEXT: return [[vsharding_annotated_1]], [[vsharding_annotated]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>
  return %sharding_annotated_1, %sharding_annotated_2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
}
