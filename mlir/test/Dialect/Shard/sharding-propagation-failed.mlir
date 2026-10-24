// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation))" %s -verify-diagnostics

shard.grid "private" @grid(shape = 1)
shard.grid "private" @grid_2(shape = 2)
// expected-error @+1 {{'func.func' op only one block is supported!}}
func.func @multi_block_function(%arg0 : tensor<6x6xi32>) -> tensor<6x6xi32> {
    %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
    %sharded = shard.shard %arg0 to %sharding : tensor<6x6xi32>
    cf.br ^bb1
  ^bb1:
    return %sharded : tensor<6x6xi32>
}

// -----

func.func @matmul_shard_broadcast_dimension(%a: tensor<1x4x8xf32>, %b: tensor<2x8x16xf32>, %zp: tensor<1xf32>) -> tensor<2x4x16xf32> {
  %sharding = shard.sharding @grid_2 split_axes = [[0]] : !shard.sharding
  %a_sharded = shard.shard %a to %sharding annotate_for_users : tensor<1x4x8xf32>
  // expected-error @+1 {{'tosa.matmul' op cannot shard operand 0 dimension 1: its indexing expression is constant}}
  %0 = tosa.matmul %a_sharded, %b, %zp, %zp : (tensor<1x4x8xf32>, tensor<2x8x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}
