// RUN: mlir-opt \
// RUN:   --verify-each \
// RUN:   --pass-pipeline="builtin.module(func.func(sharding-propagation))" \
// RUN:   %s | FileCheck %s

mesh.mesh @mesh_2_2(shape = 2)

// CHECK-LABEL: func @matmul_shard_prallel_axis
func.func @matmul_shard_prallel_axis(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<2x3xf32>,
  %arg0 : tensor<2x3xf32>,
  // CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<3x2xf32>,
  %arg1 : tensor<3x2xf32>,
  // CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<2x2xf32>
  %out_dps: tensor<2x2xf32>
) -> tensor<2x2xf32> {
  // CHECK: %[[IN1_ANNOTATED_0:.*]] = mesh.shard %[[IN1]] to <@mesh_2, {{\[}}[0]]> : tensor<2x3xf32>
  // CHECK: %[[IN1_ANNOTATED_1:.*]] = mesh.shard %[[IN1_ANNOTATED_0]] to <@mesh_2, {{\[}}[0]]> annotate_for_users : tensor<2x3xf32>
  // CHECK: %[[IN2_ANNOTATED:.*]] = mesh.shard %[[IN2]] to <@mesh_2, []> annotate_for_users : tensor<3x2xf32>
  // CHECK: %[[DPS_OUT_ANNOTATED:.*]] = mesh.shard %[[DPS_OUT]] to <@mesh_2, {{\[}}[0]]> annotate_for_users : tensor<2x2xf32>
  %arg0_sharded = mesh.shard %arg0 to <@mesh_2, [[0]]> : tensor<2x3xf32>

  // CHECK: %[[RES:.*]] = linalg.matmul ins(%[[IN1_ANNOTATED_1]], %[[IN2_ANNOTATED]] : tensor<2x3xf32>, tensor<3x2xf32>)
  // CHECK-SAME:  outs(%[[DPS_OUT_ANNOTATED]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  %res = linalg.matmul ins(%arg0_sharded, %arg1 : tensor<2x3xf32>, tensor<3x2xf32>)
    outs(%out_dps : tensor<2x2xf32>) -> tensor<2x2xf32>

  // CHECK: %[[RES_ANNOTATED_0:.*]] = mesh.shard %[[RES]] to <@mesh_2, {{\[}}[0]]> : tensor<2x2xf32>
  // CHECK: %[[RES_ANNOTATED_1:.*]] = mesh.shard %[[RES_ANNOTATED_0]] to <@mesh_2, {{\[}}[]]> annotate_for_users : tensor<2x2xf32>
  %res_sharded = mesh.shard %res to <@mesh_2, [[]]> annotate_for_users : tensor<2x2xf32>

  // CHECK: return %[[RES_ANNOTATED_1]] : tensor<2x2xf32>
  return %res_sharded : tensor<2x2xf32>
}
