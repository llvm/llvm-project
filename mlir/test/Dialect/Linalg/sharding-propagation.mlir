// RUN: mlir-opt \
// RUN:   --verify-each \
// RUN:   --pass-pipeline="builtin.module(func.func(sharding-propagation))" \
// RUN:   %s | FileCheck %s

shard.grid @grid_2(shape = 2)

// CHECK-LABEL: func @matmul_shard_prallel_axis
func.func @matmul_shard_prallel_axis(
  // CHECK-SAME: %[[IN1:[A-Za-z0-9_]+]]: tensor<2x3xf32>,
  %arg0 : tensor<2x3xf32>,
  // CHECK-SAME: %[[IN2:[A-Za-z0-9_]+]]: tensor<3x2xf32>,
  %arg1 : tensor<3x2xf32>,
  // CHECK-SAME: %[[DPS_OUT:[A-Za-z0-9_]+]]: tensor<2x2xf32>
  %out_dps: tensor<2x2xf32>
) -> tensor<2x2xf32> {
  // CHECK: %[[SIN1_ANNOTATED_0:.*]] = shard.sharding @grid_2 split_axes = {{\[}}[0]] : !shard.sharding
  // CHECK-NEXT: %[[IN1_ANNOTATED_0:.*]] = shard.shard %[[IN1]] to %[[SIN1_ANNOTATED_0]] : tensor<2x3xf32>
  // CHECK: %[[SIN1_ANNOTATED_1:.*]] = shard.sharding @grid_2 split_axes = {{\[}}[0]] : !shard.sharding
  // CHECK-NEXT: %[[IN1_ANNOTATED_1:.*]] = shard.shard %[[IN1_ANNOTATED_0]] to %[[SIN1_ANNOTATED_1]] annotate_for_users : tensor<2x3xf32>
  // CHECK: %[[SIN2_ANNOTATED:.*]] = shard.sharding @grid_2 split_axes = {{\[}}[]] : !shard.sharding
  // CHECK-NEXT: %[[IN2_ANNOTATED:.*]] = shard.shard %[[IN2]] to %[[SIN2_ANNOTATED]] annotate_for_users : tensor<3x2xf32>
  // CHECK: %[[SDPS_OUT_ANNOTATED:.*]] = shard.sharding @grid_2 split_axes = {{\[}}[0]] : !shard.sharding
  // CHECK-NEXT: %[[DPS_OUT_ANNOTATED:.*]] = shard.shard %[[DPS_OUT]] to %[[SDPS_OUT_ANNOTATED]] annotate_for_users : tensor<2x2xf32>
  %sarg0_sharded = shard.sharding @grid_2 split_axes = [[0]] : !shard.sharding
  %arg0_sharded = shard.shard %arg0 to %sarg0_sharded : tensor<2x3xf32>

  // CHECK: %[[RES:.*]] = linalg.matmul ins(%[[IN1_ANNOTATED_1]], %[[IN2_ANNOTATED]] : tensor<2x3xf32>, tensor<3x2xf32>)
  // CHECK-SAME:  outs(%[[DPS_OUT_ANNOTATED]] : tensor<2x2xf32>) -> tensor<2x2xf32>
  %res = linalg.matmul ins(%arg0_sharded, %arg1 : tensor<2x3xf32>, tensor<3x2xf32>)
    outs(%out_dps : tensor<2x2xf32>) -> tensor<2x2xf32>

  // CHECK: %[[SRES_ANNOTATED_0:.*]] = shard.sharding @grid_2 split_axes = {{\[}}[0]] : !shard.sharding
  // CHECK-NEXT: %[[RES_ANNOTATED_0:.*]] = shard.shard %[[RES]] to %[[SRES_ANNOTATED_0]] : tensor<2x2xf32>
  // CHECK: %[[SRES_ANNOTATED_1:.*]] = shard.sharding @grid_2 split_axes = {{\[}}[]] : !shard.sharding
  // CHECK-NEXT: %[[RES_ANNOTATED_1:.*]] = shard.shard %[[RES_ANNOTATED_0]] to %[[SRES_ANNOTATED_1]] annotate_for_users : tensor<2x2xf32>
  %sres_sharded = shard.sharding @grid_2 split_axes = [[]] : !shard.sharding
  %res_sharded = shard.shard %res to %sres_sharded annotate_for_users : tensor<2x2xf32>

  // CHECK: return %[[RES_ANNOTATED_1]] : tensor<2x2xf32>
  return %res_sharded : tensor<2x2xf32>
}
