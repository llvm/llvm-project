// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(shard-partition))" %s | FileCheck %s

// The input has no tensor or arith ops, so neither dialect is loaded when
// shard-partition starts. Unsharding creates a tensor.cast, so the pass has
// to declare the dialects it creates ops of.

shard.grid @grid_1d(shape = 2)

// CHECK-LABEL: func @unshard_without_tensor_ops
func.func @unshard_without_tensor_ops(
  // CHECK-SAME: %[[ARG:.*]]: tensor<5x14xf32>
  %arg0: tensor<10x14xf32>
) -> tensor<10x14xf32> {
  // CHECK: %[[ALL_GATHER:.*]] = shard.all_gather %[[ARG]] on @grid_1d grid_axes = [0] gather_axis = 0 : tensor<5x14xf32> -> tensor<10x14xf32>
  // CHECK: %[[CAST:.*]] = tensor.cast %[[ALL_GATHER]] : tensor<10x14xf32> to tensor<10x14xf32>
  %s0 = shard.sharding @grid_1d split_axes = [[0]] : !shard.sharding
  %0 = shard.shard %arg0 to %s0 : tensor<10x14xf32>
  %s1 = shard.sharding @grid_1d split_axes = [[]] : !shard.sharding
  %1 = shard.shard %0 to %s1 annotate_for_users : tensor<10x14xf32>
  // CHECK: return %[[CAST]] : tensor<10x14xf32>
  return %1 : tensor<10x14xf32>
}
