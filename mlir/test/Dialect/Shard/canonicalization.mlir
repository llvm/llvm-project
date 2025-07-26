// RUN: mlir-opt --canonicalize %s | FileCheck %s

shard.grid @grid0(shape = 2x4)

// CHECK-LABEL: func @all_reduce_empty_grid_axes
func.func @all_reduce_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.all_reduce
  %0 = shard.all_reduce %arg0 on @grid0
    grid_axes = []
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @all_reduce_empty_grid_axes_different_return_type
func.func @all_reduce_empty_grid_axes_different_return_type(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
// CHECK: shard.all_reduce
  %0 = shard.all_reduce %arg0 on @grid0
// CHECK-NOT: grid_axes
    grid_axes = []
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_reduce_default_reduction
func.func @all_reduce_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
  %0 = shard.all_reduce %arg0 on @grid0
    grid_axes = [0]
// CHECK-NOT: reduction
    reduction = sum
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @all_to_all_empty_grid_axes
func.func @all_to_all_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32>
    %arg0 : tensor<8xf32>) -> tensor<8xf32> {
// CHECK-NOT: shard.all_to_all
  %0 = shard.all_to_all %arg0 on @grid0
    grid_axes = []
    split_axis = 0
    concat_axis = 0
    : tensor<8xf32> -> tensor<8xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @all_gather_empty_grid_axes
func.func @all_gather_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.all_gather
  %0 = shard.all_gather %arg0 on @grid0
    grid_axes = []
    gather_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @all_slice_empty_grid_axes
func.func @all_slice_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.scatter
  %0 = shard.all_slice %arg0 on @grid0
    grid_axes = []
    slice_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @broadcast_empty_grid_axes
func.func @broadcast_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.broadcast
  %0 = shard.broadcast %arg0 on @grid0
    grid_axes = []
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @gather_empty_grid_axes
func.func @gather_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.gather
  %0 = shard.gather %arg0 on @grid0
    grid_axes = []
    gather_axis = 0
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @receive_empty_grid_axes
func.func @receive_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.recv
  %0 = shard.recv %arg0 on @grid0
    grid_axes = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_empty_grid_axes
func.func @reduce_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.reduce
  %0 = shard.reduce %arg0 on @grid0
    grid_axes = []
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_grid_axes
func.func @reduce_scatter_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.reduce_scatter
  %0 = shard.reduce_scatter %arg0 on @grid0
    grid_axes = []
    scatter_axis = 0
    : tensor<4xf32> -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @reduce_scatter_empty_grid_axes_different_return_type
func.func @reduce_scatter_empty_grid_axes_different_return_type(
    %arg0 : tensor<4xf32>) -> tensor<4xf64> {
// CHECK: shard.reduce_scatter
  %0 = shard.reduce_scatter %arg0 on @grid0
// CHECK-NOT: grid_axes
    grid_axes = []
    scatter_axis = 0
    : tensor<4xf32> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @reduce_scatter_default_reduction
func.func @reduce_scatter_default_reduction(
    %arg0 : tensor<4xf32>) -> tensor<2xf64> {
  %0 = shard.reduce_scatter %arg0 on @grid0
    grid_axes = [0]
// CHECK-NOT: reduction
    reduction = sum
    scatter_axis = 0
    : tensor<4xf32> -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func @scatter_empty_grid_axes
func.func @scatter_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.scatter
  %0 = shard.scatter %arg0 on @grid0
    grid_axes = []
    scatter_axis = 0
    root = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @send_empty_grid_axes
func.func @send_empty_grid_axes(
// CHECK-SAME: %[[ARG:.*]]: tensor<4xf32>
    %arg0 : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: shard.send
  %0 = shard.send %arg0 on @grid0
    grid_axes = []
    destination = []
    : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[ARG]]
  return %0 : tensor<4xf32>
}

shard.grid @grid4x4(shape = 4x4)
// CHECK-LABEL: func @test_halo_sizes
func.func @test_halo_sizes() -> !shard.sharding {
  %c2_i64 = arith.constant 2 : i64
  // CHECK shard.sharding @grid4x4 split_axes = [[0], [1]] halo_sizes = [1, 2, 2, 22] : !shard.sharding
  %sharding = shard.sharding @grid4x4 split_axes = [[0], [1]] halo_sizes = [1, %c2_i64, %c2_i64, 22] : !shard.sharding
  return %sharding : !shard.sharding
}

// CHECK-LABEL: func @test_shard_offs
func.func @test_shard_offs() -> !shard.sharding {
  %c2_i64 = arith.constant 2 : i64
  // CHECK shard.sharding @grid4x4 split_axes = [[0], [1]] sharded_dims_offsets = [0, 1, 2, 3, 4, 0, 2, 3, 4, 22] : !shard.sharding
  %sharding = shard.sharding @grid4x4 split_axes = [[0], [1]] sharded_dims_offsets = [0, 1, %c2_i64, 3, 4, 0, %c2_i64, 3, 4, 22] : !shard.sharding
  return %sharding : !shard.sharding
}

// CHECK-LABEL: func @test_duplicate_shardops
func.func @test_duplicate_shardops() -> (tensor<1024x1024xf32>, tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0, 1]] : !shard.sharding
  %sharding_1 = shard.sharding @grid4x4 split_axes = [[0, 1]] : !shard.sharding
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_2 = shard.sharding @grid4x4 split_axes = [[0, 1]] : !shard.sharding
  %sharded_2 = shard.shard %cst_2 to %sharding_2 : tensor<1024x1024xf32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_3 = shard.sharding @grid4x4 split_axes = [[0, 1]] : !shard.sharding
  %sharded_3 = shard.shard %cst_3 to %sharding_3 : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharded:%.*]] = shard.shard [[vcst]] to [[vsharding]] : tensor<1024x1024xf32>
  %sharded_1 = shard.shard %cst_1 to %sharding_1 : tensor<1024x1024xf32>
  // CHECK-NEXT: return [[vsharded]], [[vsharded]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>
  return %sharded_1, %sharded_2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
}

// CHECK-LABEL: func @test_duplicate_shardops_diff
func.func @test_duplicate_shardops_diff() -> (tensor<1024x1024xf32>, tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
  %sharding_1 = shard.sharding @grid4x4 split_axes = [[0]] : !shard.sharding
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharding_0:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0, 1]] : !shard.sharding
  %sharding_2 = shard.sharding @grid4x4 split_axes = [[0, 1]] : !shard.sharding
  // CHECK-NEXT: [[vsharded:%.*]] = shard.shard [[vcst]] to [[vsharding_0]] : tensor<1024x1024xf32>
  %sharded_2 = shard.shard %cst_2 to %sharding_2 : tensor<1024x1024xf32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
  %sharding_3 = shard.sharding @grid4x4 split_axes = [[0]] : !shard.sharding
  %sharded_3 = shard.shard %cst_3 to %sharding_3 : tensor<1024x1024xf32>
  // CHECK-NEXT: [[vsharded_1:%.*]] = shard.shard [[vsharded]] to [[vsharding]] : tensor<1024x1024xf32>
  %sharded_1 = shard.shard %cst_1 to %sharding_1 : tensor<1024x1024xf32>
  // CHECK-NEXT: return [[vsharded_1]], [[vsharded]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>
  return %sharded_1, %sharded_2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>
}
