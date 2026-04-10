// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation))" %s -verify-diagnostics

shard.grid @grid(shape = 1) {sym_visibility = "private"}
// expected-error @+1 {{'func.func' op only one block is supported!}}
func.func @multi_block_function(%arg0 : tensor<6x6xi32>) -> tensor<6x6xi32> {
    %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
    %sharded = shard.shard %arg0 to %sharding : tensor<6x6xi32>
    cf.br ^bb1
  ^bb1:
    return %sharded : tensor<6x6xi32>
}
