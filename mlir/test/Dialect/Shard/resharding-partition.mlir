// RUN: mlir-opt -test-grid-resharding-partition %s | FileCheck %s

shard.grid @grid_1d(shape = 2)
shard.grid @grid_1d_dynamic(shape = ?)

// CHECK-LABEL: func @same_source_and_target_sharding
func.func @same_source_and_target_sharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xf32>
  %arg0: tensor<2xf32>
) -> tensor<2xf32> {
  %s0 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<2xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<2xf32>
  // CHECK: return %[[ARG]]
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func @identical_source_and_target_sharding
func.func @identical_source_and_target_sharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xf32>
  %arg0: tensor<2xf32>
) -> tensor<2xf32> {
  %s0 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<2xf32>
  %1 = shard.shard %0 to %s0 annotate_for_users : tensor<2xf32>
  // CHECK: return %[[ARG]]
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func @split_replicated_tensor_axis
func.func @split_replicated_tensor_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<3x14xf32>
  %arg0: tensor<3x14xf32>
) -> tensor<3x14xf32> {
  // CHECK: %[[ALL_SLICE:.*]] = shard.all_slice %[[ARG]] on @grid_1d grid_axes = [0] slice_axis = 1
  // CHECK-SAME: tensor<3x14xf32> -> tensor<3x7xf32>
  // CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[ALL_SLICE]] : tensor<3x7xf32> to tensor<3x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<3x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[], [0]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<3x14xf32>
  // CHECK: return %[[RESULT]] : tensor<3x14xf32>
  return %1 : tensor<3x14xf32>
}

// CHECK-LABEL: func @split_replicated_tensor_axis_dynamic
func.func @split_replicated_tensor_axis_dynamic(
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x3x?xf32>
  %arg0: tensor<?x3x?xf32>
) -> tensor<?x3x?xf32> {
  // CHECK: %[[RESULT:.*]] = shard.all_slice %[[ARG]] on @grid_1d_dynamic grid_axes = [0] slice_axis = 0
  // CHECK-SAME: tensor<?x3x?xf32> -> tensor<?x3x?xf32>
  %s0 = shard.sharding @grid_1d_dynamic split_axes = [[], [], []] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<?x3x?xf32>
  %s1 = shard.sharding @grid_1d_dynamic split_axes = [[0]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<?x3x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x3x?xf32>
  return %1 : tensor<?x3x?xf32>
}

// CHECK-LABEL: func @move_split_axis
func.func @move_split_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<5x14xf32>
  // CHECK: %[[TARGET_SHARD:.*]] = shard.all_to_all %[[SOURCE_SHARD]] on @grid_1d grid_axes = [0] split_axis = 1 concat_axis = 0 : tensor<5x14xf32> -> tensor<10x7xf32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[TARGET_SHARD]] : tensor<10x7xf32> to tensor<10x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<10x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[], [0]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[RES]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @move_split_axis_dynamic_grid
func.func @move_split_axis_dynamic_grid(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<?x14xf32>
  // CHECK: %[[ALL_TO_ALL:.*]] = shard.all_to_all %[[SOURCE_SHARD]] on @grid_1d_dynamic grid_axes = [0] split_axis = 1 concat_axis = 0 : tensor<?x14xf32> -> tensor<?x?xf32>
  // CHECK: %[[TARGET_SHARD:.*]] = tensor.cast %[[ALL_TO_ALL]] : tensor<?x?xf32> to tensor<10x?xf32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[TARGET_SHARD]] : tensor<10x?xf32> to tensor<10x14xf32>
  %s0 = shard.sharding @grid_1d_dynamic split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<10x14xf32>
  %s1 = shard.sharding @grid_1d_dynamic split_axes = [[], [0]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[RES]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @move_split_dynamic_axis
func.func @move_split_dynamic_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x14xf32>
  %arg0: tensor<?x14xf32>
) -> tensor<?x14xf32> {
  // CHECK: %[[TARGET_SHARD:.*]] = shard.all_to_all %[[ARG]] on @grid_1d grid_axes = [0] split_axis = 1 concat_axis = 0 : tensor<?x14xf32> -> tensor<?x7xf32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[TARGET_SHARD]] : tensor<?x7xf32> to tensor<?x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<?x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[], [0]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<?x14xf32>
  // CHECK: return %[[RES]] : tensor<?x14xf32>
  return %1 : tensor<?x14xf32>
}

// CHECK-LABEL: func @unshard_static_axis
func.func @unshard_static_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<5x14xf32>
  // CHECK: %[[ALL_GATHER:.*]] = shard.all_gather %[[SOURCE_SHARD]] on @grid_1d grid_axes = [0] gather_axis = 0 : tensor<5x14xf32> -> tensor<10x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<10x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @unshard_static_last_axis
func.func @unshard_static_last_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<10x7xf32>
  // CHECK: %[[ALL_GATHER:.*]] = shard.all_gather %[[SOURCE_SHARD]] on @grid_1d grid_axes = [0] gather_axis = 1 : tensor<10x7xf32> -> tensor<10x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[], [0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<10x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[], []] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @unshard_dynamic_axis
func.func @unshard_dynamic_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x14xf32>
  %arg0: tensor<?x14xf32>
) -> tensor<?x14xf32> {
  // CHECK: %[[ALL_GATHER:.*]] = shard.all_gather %[[ARG]] on @grid_1d grid_axes = [0] gather_axis = 0 : tensor<?x14xf32> -> tensor<?x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<?x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<?x14xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<?x14xf32>
  return %1 : tensor<?x14xf32>
}

// CHECK-LABEL: func @unshard_static_axis_on_dynamic_grid_axis
func.func @unshard_static_axis_on_dynamic_grid_axis(
// CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>  
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<?x14xf32>
  // CHECK: %[[ALL_GATHER:.*]] = shard.all_gather %[[SOURCE_SHARD]] on @grid_1d_dynamic grid_axes = [0] gather_axis = 0 : tensor<?x14xf32> -> tensor<?x14xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ALL_GATHER]] : tensor<?x14xf32> to tensor<10x14xf32>
  %s0 = shard.sharding @grid_1d_dynamic split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<10x14xf32>
  %s1 = shard.sharding @grid_1d_dynamic split_axes = [[]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[RES]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}
