// RUN: mlir-opt -test-mesh-resharding-spmdization %s | FileCheck %s

mesh.mesh @mesh_1d(shape = 2)
mesh.mesh @mesh_1d_dynamic(shape = ?)

// CHECK-LABEL: func @same_source_and_target_sharding
func.func @same_source_and_target_sharding(
  // CHECK-SAME: %[[ARG:.*]]: tensor<2xf32>
  %arg0: tensor<2xf32>
) -> tensor<2xf32> {
  %0 = mesh.shard %arg0 to <@mesh_1d, [[]]> : tensor<2xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[]]> annotate_for_users : tensor<2xf32>
  // CHECK: return %[[ARG]]
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func @split_replicated_tensor_axis
func.func @split_replicated_tensor_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<3x14xf32>
  %arg0: tensor<3x14xf32>
) -> tensor<3x14xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[TENSOR_SPLIT_AXIS_SIZE:.*]] = arith.constant 14 : index
  // CHECK: %[[PROCESS_INDEX:.*]] = mesh.process_multi_index on @mesh_1d axes = [0] : index
  // CHECK: %[[MESH_AXIS_SIZE:.*]] = mesh.mesh_shape @mesh_1d axes = [0] : index
  // CHECK: %[[TENSOR_SPLIT_AXIS_SIZE_MOD_MESH_AXIS_SIZE:.*]] = arith.remui %[[TENSOR_SPLIT_AXIS_SIZE]], %[[MESH_AXIS_SIZE]] : index
  // CHECK: %[[RESULT_TENSOR_AXIS_SIZE_CHECK:.*]] = arith.cmpi eq, %[[TENSOR_SPLIT_AXIS_SIZE_MOD_MESH_AXIS_SIZE]], %[[ZERO]] : index
  // CHECK: cf.assert %[[RESULT_TENSOR_AXIS_SIZE_CHECK]]
  // CHECK: %[[RESULT_TENSOR_AXIS_SIZE:.*]] = arith.divui %[[TENSOR_SPLIT_AXIS_SIZE]], %[[MESH_AXIS_SIZE]] : index
  // CHECK: %[[RESULT_TENSOR_AXIS_OFFSET:.*]] = arith.muli %[[RESULT_TENSOR_AXIS_SIZE]], %[[PROCESS_INDEX]] : index
  // CHECK: %[[RESULT_TENSOR_SLICE:.*]] = tensor.extract_slice %[[ARG]][0, %[[RESULT_TENSOR_AXIS_OFFSET]]] [3, 7] [1, 1] : tensor<3x14xf32> to tensor<3x7xf32>
  // CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_TENSOR_SLICE]] : tensor<3x7xf32> to tensor<3x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[]]> : tensor<3x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[], [0]]> annotate_for_users : tensor<3x14xf32>
  // CHECK: return %[[RESULT]] : tensor<3x14xf32>
  return %1 : tensor<3x14xf32>
}

// CHECK-LABEL: func @split_replicated_tensor_axis_dynamic
func.func @split_replicated_tensor_axis_dynamic(
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x3x?xf32>
  %arg0: tensor<?x3x?xf32>
) -> tensor<?x3x?xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
  // CHECK: %[[PROCESS_INDEX:.*]] = mesh.process_multi_index on @mesh_1d_dynamic axes = [0] : index
  // CHECK: %[[MESH_AXIS_SIZE:.*]] = mesh.mesh_shape @mesh_1d_dynamic axes = [0] : index
  // CHECK: %[[TENSOR_SPLIT_AXIS_SIZE:.*]] = tensor.dim %[[ARG]], %[[ZERO]] : tensor<?x3x?xf32>
  // CHECK: %[[TENSOR_SPLIT_AXIS_SIZE_MOD_MESH_AXIS_SIZE:.*]] = arith.remui %[[TENSOR_SPLIT_AXIS_SIZE]], %[[MESH_AXIS_SIZE]] : index
  // CHECK: %[[RESULT_TENSOR_AXIS_SIZE_CHECK:.*]] = arith.cmpi eq, %[[TENSOR_SPLIT_AXIS_SIZE_MOD_MESH_AXIS_SIZE]], %[[ZERO]] : index
  // CHECK: cf.assert %[[RESULT_TENSOR_AXIS_SIZE_CHECK]]
  // CHECK: %[[RESULT_TENSOR_SPLIT_AXIS_SIZE:.*]] = arith.divui %[[TENSOR_SPLIT_AXIS_SIZE]], %[[MESH_AXIS_SIZE]] : index
  // CHECK: %[[RESULT_TENSOR_SPLIT_AXIS_OFFSET:.*]] = arith.muli %[[RESULT_TENSOR_SPLIT_AXIS_SIZE]], %[[PROCESS_INDEX]] : index
  // CHECK: %[[TENSOR_AXIS_2_SIZE:.*]] = tensor.dim %[[ARG]], %[[TWO]] : tensor<?x3x?xf32>
  // CHECK: %[[RESULT_TENSOR_SLICE:.*]] = tensor.extract_slice %[[ARG]][%[[RESULT_TENSOR_SPLIT_AXIS_OFFSET]], 0, 0]
  // CHECK-SAME: [%[[RESULT_TENSOR_SPLIT_AXIS_SIZE]], 3, %[[TENSOR_AXIS_2_SIZE]]] [1, 1, 1] : tensor<?x3x?xf32> to tensor<?x3x?xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d_dynamic, [[], [], []]> : tensor<?x3x?xf32>
  %1 = mesh.shard %0 to <@mesh_1d_dynamic, [[0]]> annotate_for_users : tensor<?x3x?xf32>
  // CHECK: return %[[RESULT_TENSOR_SLICE]] : tensor<?x3x?xf32>
  return %1 : tensor<?x3x?xf32>
}

// CHECK-LABEL: func @move_split_axis
func.func @move_split_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<5x14xf32>
  // CHECK: %[[TARGET_SHARD:.*]] = mesh.all_to_all %[[SOURCE_SHARD]] on @mesh_1d mesh_axes = [0] split_axis = 1 concat_axis = 0 : tensor<5x14xf32> -> tensor<10x7xf32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[TARGET_SHARD]] : tensor<10x7xf32> to tensor<10x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<10x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[], [0]]> annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[RES]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @move_split_axis_dynamic_mesh
func.func @move_split_axis_dynamic_mesh(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<?x14xf32>
  // CHECK: %[[ALL_TO_ALL:.*]] = mesh.all_to_all %[[SOURCE_SHARD]] on @mesh_1d_dynamic mesh_axes = [0] split_axis = 1 concat_axis = 0 : tensor<?x14xf32> -> tensor<?x?xf32>
  // CHECK: %[[TARGET_SHARD:.*]] = tensor.cast %[[ALL_TO_ALL]] : tensor<?x?xf32> to tensor<10x?xf32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[TARGET_SHARD]] : tensor<10x?xf32> to tensor<10x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d_dynamic, [[0]]> : tensor<10x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d_dynamic, [[], [0]]> annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[RES]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @move_split_dynamic_axis
func.func @move_split_dynamic_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x14xf32>
  %arg0: tensor<?x14xf32>
) -> tensor<?x14xf32> {
  // CHECK: %[[TARGET_SHARD:.*]] = mesh.all_to_all %[[ARG]] on @mesh_1d mesh_axes = [0] split_axis = 1 concat_axis = 0 : tensor<?x14xf32> -> tensor<?x7xf32>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[TARGET_SHARD]] : tensor<?x7xf32> to tensor<?x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<?x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[], [0]]> annotate_for_users : tensor<?x14xf32>
  // CHECK: return %[[RES]] : tensor<?x14xf32>
  return %1 : tensor<?x14xf32>
}

// CHECK-LABEL: func @unshard_static_axis
func.func @unshard_static_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<5x14xf32>
  // CHECK: %[[ALL_GATHER:.*]] = mesh.all_gather %[[SOURCE_SHARD]] on @mesh_1d mesh_axes = [0] gather_axis = 0 : tensor<5x14xf32> -> tensor<10x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<10x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[]]> annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @unshard_dynamic_axis
func.func @unshard_dynamic_axis(
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x14xf32>
  %arg0: tensor<?x14xf32>
) -> tensor<?x14xf32> {
  // CHECK: %[[ALL_GATHER:.*]] = mesh.all_gather %[[ARG]] on @mesh_1d mesh_axes = [0] gather_axis = 0 : tensor<?x14xf32> -> tensor<?x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<?x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[]]> annotate_for_users : tensor<?x14xf32>
  // CHECK: return %[[ALL_GATHER]] : tensor<?x14xf32>
  return %1 : tensor<?x14xf32>
}

// CHECK-LABEL: func @unshard_static_axis_on_dynamic_mesh_axis
func.func @unshard_static_axis_on_dynamic_mesh_axis(
// CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>  
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[SOURCE_SHARD:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : tensor<10x14xf32> to tensor<?x14xf32>
  // CHECK: %[[ALL_GATHER:.*]] = mesh.all_gather %[[SOURCE_SHARD]] on @mesh_1d_dynamic mesh_axes = [0] gather_axis = 0 : tensor<?x14xf32> -> tensor<?x14xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ALL_GATHER]] : tensor<?x14xf32> to tensor<10x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d_dynamic, [[0]]> : tensor<10x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d_dynamic, [[]]> annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[RES]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}

// CHECK-LABEL: func @partial_axis
func.func @partial_axis(
// CHECK-SAME: %[[ARG:.*]]: tensor<10x14xf32>  
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[ALL_REDUCE:.*]] = mesh.all_reduce %[[ARG]] on @mesh_1d mesh_axes = [0] : tensor<10x14xf32> -> tensor<10x14xf32>
  %0 = mesh.shard %arg0 to <@mesh_1d, [[]], partial = sum[0]> : tensor<10x14xf32>
  %1 = mesh.shard %0 to <@mesh_1d, [[]]> annotate_for_users : tensor<10x14xf32>
  // CHECK: %[[ALL_REDUCE]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}
