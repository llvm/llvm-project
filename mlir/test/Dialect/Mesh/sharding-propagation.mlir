// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation,cse))" %s | FileCheck %s

mesh.mesh @mesh_2(shape = 2)
mesh.mesh @mesh_1d(shape = ?)
mesh.mesh @mesh_2d(shape = 2x4)
mesh.mesh @mesh_3d(shape = ?x?x?)

// CHECK-LABEL: func.func @element_wise_empty_sharding_info
func.func @element_wise_empty_sharding_info(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: tosa.sigmoid
  %0 = tosa.sigmoid %arg0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @element_wise_on_def
// CHECK-SAME:    %[[ARG:.*]]: tensor<8x16xf32>
func.func @element_wise_on_def(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V1:.*]] = tosa.sigmoid %[[V0]]
  %0 = tosa.sigmoid %arg0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:  %[[V2:.*]] = mesh.shard %[[V1]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  %1 = mesh.shard %0 to <@mesh_2d, [[0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT:  return %[[V2]]
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @element_wise_on_use
// CHECK-SAME:    %[[ARG:.*]]: tensor<8x16xf32>
func.func @element_wise_on_use(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  %0 = mesh.shard %arg0 to <@mesh_2d, [[0], [1]]> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V1:.*]] = tosa.sigmoid %[[V0]]
  %1 = tosa.sigmoid %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:  %[[V2:.*]] = mesh.shard %[[V1]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT:  return %[[V2]]
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @element_wise_on_graph_output
// CHECK-SAME:    %[[ARG:.*]]: tensor<8x16xf32>
func.func @element_wise_on_graph_output(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V1:.*]] = tosa.sigmoid %[[V0]]
  %0 = tosa.sigmoid %arg0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:  %[[V2:.*]] = mesh.shard %[[V1]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  %1 = mesh.shard %0 to <@mesh_2d, [[0], [1]]> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT:  return %[[V3]]
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @element_wise_on_graph_input
// CHECK-SAME:    %[[ARG:.*]]: tensor<8x16xf32>
func.func @element_wise_on_graph_input(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V1:.*]] = mesh.shard %[[V0]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  %0 = mesh.shard %arg0 to <@mesh_2d, [[0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V2:.*]] = tosa.sigmoid %[[V1]]
  %1 = tosa.sigmoid %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT:  return %[[V3]]
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @arrow_structure
// CHECK-SAME:    %[[ARG:.*]]: tensor<8x16xf32>
func.func @arrow_structure(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT:  %[[V1:.*]] = mesh.shard %[[ARG]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V2:.*]] = tosa.tanh %[[V1]]
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  %0 = tosa.tanh %arg0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:  %[[V4:.*]] = mesh.shard %[[V3]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<8x16xf32>
  // CHECK-NEXT:  %[[V5:.*]] = tosa.abs %[[V4]]
  // CHECK-NEXT:  %[[V6:.*]] = mesh.shard %[[V5]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  %1 = tosa.abs %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:  %[[V7:.*]] = tosa.negate %[[V4]]
  // CHECK-NEXT:  %[[V8:.*]] = mesh.shard %[[V7]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<8x16xf32>
  %2 = tosa.negate %0 : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %3 = mesh.shard %2 to <@mesh_2d, [[0], [1]]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[V6]], %[[V8]]
  return %1, %3 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func.func @matmul_on_def_shard_batch_and_m
// CHECK-SAME:     %[[ARG0:.*]]: tensor<2x16x8xf32>, %[[ARG1:.*]]: tensor<2x8x32xf32>
func.func @matmul_on_def_shard_batch_and_m(%arg0: tensor<2x16x8xf32>, %arg1: tensor<2x8x32xf32>) -> tensor<2x16x32xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG0]] to <@mesh_2d, {{\[\[}}0], [1]]> annotate_for_users : tensor<2x16x8xf32>
  // CHECK-NEXT:  %[[V1:.*]] = mesh.shard %[[ARG1]] to <@mesh_2d, {{\[\[}}0]]> annotate_for_users : tensor<2x8x32xf32>
  // CHECK-NEXT:  %[[V2:.*]] = tosa.matmul %[[V0]], %[[V1]]
  %0 = tosa.matmul %arg0, %arg1 : (tensor<2x16x8xf32>, tensor<2x8x32xf32>) -> tensor<2x16x32xf32>
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}0], [1]]> : tensor<2x16x32xf32>
  %1 = mesh.shard %0 to <@mesh_2d, [[0], [1]]> : tensor<2x16x32xf32>
  // CHECK-NEXT:  return %[[V3]]
  return %1 : tensor<2x16x32xf32>
}

// CHECK-LABEL: func.func @matmul_on_def_shard_m_and_k
// CHECK-SAME:     %[[ARG0:.*]]: tensor<2x16x8xf32>, %[[ARG1:.*]]: tensor<2x8x32xf32>
func.func @matmul_on_def_shard_m_and_k(%arg0: tensor<2x16x8xf32>, %arg1: tensor<2x8x32xf32>) -> tensor<2x16x32xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG0]] to <@mesh_2d, {{\[\[}}], [1], [0]]> annotate_for_users : tensor<2x16x8xf32>
  // CHECK-NEXT:  %[[V1:.*]] = mesh.shard %[[ARG1]] to <@mesh_2d, {{\[\[}}], [0]]> annotate_for_users : tensor<2x8x32xf32>
  // CHECK-NEXT:  %[[V2:.*]] = tosa.matmul %[[V0]], %[[V1]]
  %0 = tosa.matmul %arg0, %arg1 : (tensor<2x16x8xf32>, tensor<2x8x32xf32>) -> tensor<2x16x32xf32>
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}], [1]], partial = sum[0]> : tensor<2x16x32xf32>
  %1 = mesh.shard %0 to <@mesh_2d, [[], [1]], partial = sum[0]> : tensor<2x16x32xf32>
  // CHECK-NEXT:  return %[[V3]]
  return %1 : tensor<2x16x32xf32>
}

// CHECK-LABEL: func.func @matmul_on_use_shard_m_and_k
// CHECK-SAME:     %[[ARG0:.*]]: tensor<2x16x8xf32>, %[[ARG1:.*]]: tensor<2x8x32xf32>
func.func @matmul_on_use_shard_m_and_k(%arg0: tensor<2x16x8xf32>, %arg1: tensor<2x8x32xf32>) -> tensor<2x16x32xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG0]] to <@mesh_2d, {{\[\[}}], [1], [0]]> annotate_for_users : tensor<2x16x8xf32>
  %0 = mesh.shard %arg0 to <@mesh_2d, [[], [1], [0]]> annotate_for_users : tensor<2x16x8xf32>
  // CHECK-NEXT:  %[[V1:.*]] = mesh.shard %[[ARG1]] to <@mesh_2d, {{\[\[}}], [0]]> annotate_for_users : tensor<2x8x32xf32>
  // CHECK-NEXT:  %[[V2:.*]] = tosa.matmul %[[V0]], %[[V1]]
  %1 = tosa.matmul %0, %arg1 : (tensor<2x16x8xf32>, tensor<2x8x32xf32>) -> tensor<2x16x32xf32>
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}], [1]], partial = sum[0]> : tensor<2x16x32xf32>
  // CHECK-NEXT:  return %[[V3]]
  return %1 : tensor<2x16x32xf32>
}

// CHECK-LABEL: func.func @matmul_on_use_shard_m_and_duplicted_k
// CHECK-SAME:     %[[ARG0:.*]]: tensor<2x16x8xf32>, %[[ARG1:.*]]: tensor<2x8x32xf32>
func.func @matmul_on_use_shard_m_and_duplicted_k(%arg0: tensor<2x16x8xf32>, %arg1: tensor<2x8x32xf32>) -> tensor<2x16x32xf32> {
  // CHECK-NEXT:  %[[V0:.*]] = mesh.shard %[[ARG0]] to <@mesh_2d, {{\[\[}}], [1], [0]]> annotate_for_users : tensor<2x16x8xf32>
  %0 = mesh.shard %arg0 to <@mesh_2d, [[], [1], [0]]> annotate_for_users : tensor<2x16x8xf32>
  // CHECK-NEXT:  %[[V1:.*]] = mesh.shard %[[ARG1]] to <@mesh_2d, {{\[\[}}], [0]]> annotate_for_users : tensor<2x8x32xf32>
  %1 = mesh.shard %arg1 to <@mesh_2d, [[], [0]]> annotate_for_users : tensor<2x8x32xf32>
  // CHECK-NEXT:  %[[V2:.*]] = tosa.matmul %[[V0]], %[[V1]]
  %2 = tosa.matmul %0, %1 : (tensor<2x16x8xf32>, tensor<2x8x32xf32>) -> tensor<2x16x32xf32>
  // CHECK-NEXT:  %[[V3:.*]] = mesh.shard %[[V2]] to <@mesh_2d, {{\[\[}}], [1]], partial = sum[0]> : tensor<2x16x32xf32>
  // CHECK-NEXT:  return %[[V3]]
  return %2 : tensor<2x16x32xf32>
}

// CHECK-LABEL: func.func @resolve_conflicting_annotations
func.func @resolve_conflicting_annotations(
  // CHECK-SAME: %[[IN1:.*]]: tensor<2x3xf32>,
  %arg0: tensor<2x3xf32>,
  // CHECK-SAME: %[[IN2:.*]]: tensor<3x2xf32>,
  %arg1: tensor<3x2xf32>,
  // CHECK-SAME: %[[OUT_DPS:.*]]: tensor<2x2xf32>
  %out_dps: tensor<2x2xf32>
// CHECK-SAME: ) -> tensor<2x2xf32> {
) -> tensor<2x2xf32> {
  // CHECK: %[[IN1_SHARDED1:.*]] = mesh.shard %[[IN1]] to <@mesh_2, {{\[\[}}0]]> : tensor<2x3xf32>
  // CHECK: %[[IN1_SHARDED2:.*]] = mesh.shard %[[IN1_SHARDED1]] to <@mesh_2, {{\[}}]> annotate_for_users : tensor<2x3xf32>
  // CHECK: %[[IN2_SHARDED:.*]] = mesh.shard %[[IN2]] to <@mesh_2, []> annotate_for_users : tensor<3x2xf32>
  // CHECK: %[[OUT_DPS_SHARDED:.*]] = mesh.shard %[[OUT_DPS]] to <@mesh_2, {{\[}}]> annotate_for_users : tensor<2x2xf32>
  %arg0_sharded = mesh.shard %arg0 to <@mesh_2, [[0]]> : tensor<2x3xf32>

  // CHECK: %[[MATMUL:.*]] = linalg.matmul ins(%[[IN1_SHARDED2]], %[[IN2_SHARDED]] : tensor<2x3xf32>, tensor<3x2xf32>)
  // CHECK-SAME: outs(%[[OUT_DPS_SHARDED]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  %res = linalg.matmul ins(%arg0_sharded, %arg1 : tensor<2x3xf32>, tensor<3x2xf32>)
    outs(%out_dps : tensor<2x2xf32>) -> tensor<2x2xf32>

  // CHECK: %[[MATMUL_SHARDED1:.*]] = mesh.shard %[[MATMUL]] to <@mesh_2, {{\[\[}}]]> : tensor<2x2xf32>
  %res_sharded = mesh.shard %res to <@mesh_2, [[]]> : tensor<2x2xf32>

  // CHECK: return %[[MATMUL_SHARDED1]] : tensor<2x2xf32>
  return %res_sharded : tensor<2x2xf32>
}

// https://arxiv.org/abs/2211.05102 Figure 2(a)
// CHECK-LABEL: func.func @mlp_1d_weight_stationary
// CHECK-SAME:     %[[ARG0:.*]]: tensor<2x4x8xf32>, %[[ARG1:.*]]: tensor<2x8x32xf32>, %[[ARG2:.*]]: tensor<2x32x8xf32>
func.func @mlp_1d_weight_stationary(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x32xf32>, %arg2: tensor<2x32x8xf32>) -> tensor<2x4x8xf32> {
  %0 = mesh.shard %arg0 to <@mesh_1d, [[], [], [0]]> : tensor<2x4x8xf32>
  // CHECK: %[[V0:.*]] = tosa.matmul
  %1 = tosa.matmul %0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x32xf32>) -> tensor<2x4x32xf32>
  // CHECK-DAG: %[[V1:.*]] = mesh.shard %[[V0]] to <@mesh_1d, {{\[\[}}], [], [0]]> : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V2:.*]] = mesh.shard %[[V1]] to <@mesh_1d, {{\[\[}}], [], [0]]> annotate_for_users : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V3:.*]] = tosa.sigmoid %[[V2]]
  %2 = tosa.sigmoid %1 : (tensor<2x4x32xf32>) -> tensor<2x4x32xf32>
  // CHECK-DAG: %[[V4:.*]] = mesh.shard %[[V3]] to <@mesh_1d, {{\[\[}}], [], [0]]> : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V5:.*]] = mesh.shard %[[V4]] to <@mesh_1d, {{\[\[}}], [], [0]]> annotate_for_users : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V6:.*]] = mesh.shard %[[ARG2]] to <@mesh_1d, {{\[\[}}], [0]]> annotate_for_users : tensor<2x32x8xf32>
  // CHECK-DAG: %[[V7:.*]] = tosa.matmul %[[V5]], %[[V6]]
  %3 = tosa.matmul %2, %arg2 : (tensor<2x4x32xf32>, tensor<2x32x8xf32>) -> tensor<2x4x8xf32>
  // CHECK-DAG: %[[V8:.*]] = mesh.shard %[[V7]] to <@mesh_1d, {{\[\[}}], [], []], partial = sum[0]> : tensor<2x4x8xf32>
  %4 = mesh.shard %3 to <@mesh_1d, [[], [], []], partial = sum[0]> : tensor<2x4x8xf32>
  // CHECK-DAG: %[[V9:.*]] = mesh.shard %[[V8]] to <@mesh_1d, {{\[\[}}], [], [0]]> annotate_for_users : tensor<2x4x8xf32>
  %5 = mesh.shard %4 to <@mesh_1d, [[], [], [0]]> annotate_for_users : tensor<2x4x8xf32>
  // CHECK-DAG: return %[[V9]]
  return %5 : tensor<2x4x8xf32>
}

// https://arxiv.org/abs/2211.05102 Figure 2(b)
// CHECK-LABEL: func.func @mlp_2d_weight_stationary
// CHECK-SAME:     %[[ARG0:.*]]: tensor<2x4x8xf32>, %[[ARG1:.*]]: tensor<2x8x32xf32>, %[[ARG2:.*]]: tensor<2x32x8xf32>
func.func @mlp_2d_weight_stationary(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x32xf32>, %arg2: tensor<2x32x8xf32>) -> tensor<2x4x8xf32> {
  // CHECK-DAG: %[[V0:.*]] = mesh.shard %[[ARG0]] to <@mesh_3d, {{\[\[}}], [], [0, 1, 2]]> : tensor<2x4x8xf32>
  %0 = mesh.shard %arg0 to <@mesh_3d, [[], [], [0, 1, 2]]> : tensor<2x4x8xf32>
  // CHECK-DAG: %[[V1:.*]] = mesh.shard %[[V0]] to <@mesh_3d, {{\[\[}}], [], [0]]> annotate_for_users : tensor<2x4x8xf32>
  // CHECK-DAG: %[[V2:.*]] = mesh.shard %[[ARG1]] to <@mesh_3d, {{\[\[}}], [0], [1, 2]]> annotate_for_users : tensor<2x8x32xf32>
  // CHECK-DAG: %[[V3:.*]] = tosa.matmul %[[V1]], %[[V2]]
  %1 = tosa.matmul %0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x32xf32>) -> tensor<2x4x32xf32>
  // CHECK-DAG: %[[V4:.*]] = mesh.shard %[[V3]] to <@mesh_3d,  {{\[\[}}], [], [1, 2]], partial = sum[0]> : tensor<2x4x32xf32>
  %2 = mesh.shard %1 to <@mesh_3d, [[], [], [1, 2]], partial = sum[0]> : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V5:.*]] = mesh.shard %[[V4]] to <@mesh_3d, {{\[\[}}], [], [1, 2]]> annotate_for_users : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V6:.*]] = tosa.sigmoid %[[V5]]
  %3 = tosa.sigmoid %2 : (tensor<2x4x32xf32>) -> tensor<2x4x32xf32>
  // CHECK-DAG: %[[V7:.*]] = mesh.shard %[[V6]] to <@mesh_3d, {{\[\[}}], [], [1, 2]]> : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V8:.*]] = mesh.shard %[[V7]] to <@mesh_3d, {{\[\[}}], [], [1, 2]]> annotate_for_users : tensor<2x4x32xf32>
  // CHECK-DAG: %[[V9:.*]] = mesh.shard %[[ARG2]] to <@mesh_3d, {{\[\[}}], [1, 2], [0]]> annotate_for_users : tensor<2x32x8xf32>
  // CHECK-DAG: %[[V10:.*]] = tosa.matmul %[[V8]], %[[V9]]
  %4 = tosa.matmul %3, %arg2 : (tensor<2x4x32xf32>, tensor<2x32x8xf32>) -> tensor<2x4x8xf32>
  // CHECK-DAG: %[[V11:.*]] = mesh.shard %[[V10]] to <@mesh_3d, {{\[\[}}], [], [0]], partial = sum[1, 2]> : tensor<2x4x8xf32>
  %5 = mesh.shard %4 to <@mesh_3d, [[], [], [0]], partial = sum[1, 2]> : tensor<2x4x8xf32>
  // CHECK-DAG: %[[V12:.*]] = mesh.shard %[[V11]] to <@mesh_3d, {{\[\[}}], [], [0, 1, 2]]> annotate_for_users : tensor<2x4x8xf32>
  %6 = mesh.shard %5 to <@mesh_3d, [[], [], [0, 1, 2]]> annotate_for_users : tensor<2x4x8xf32>
  // CHECK-DAG: return %[[V12]]
  return %6 : tensor<2x4x8xf32>
}
