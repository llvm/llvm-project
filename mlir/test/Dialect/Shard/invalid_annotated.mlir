// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(shard-partition,test-single-fold))" \
// RUN:   -verify-diagnostics %s

shard.grid @grid(shape = 2)

// expected-error @+1 {{Cannot partition: expected a shard.shard op for block argument 0 in block 0}}
func.func @test_block_arg_missing_shard(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %2 = tosa.abs %arg0 : (tensor<6xi32>) -> tensor<6xi32>
  %sharded = shard.shard %2 to %sharding annotate_for_users : tensor<6xi32>
  return %2 : tensor<6xi32>
}

func.func @test_operand_missing_annotate(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %2 = shard.shard %arg0 to %sharding : tensor<6xi32>
  // expected-error @+1 {{Cannot partition: shard.shard for operand 0 must set 'annotate_for_users'.}}
  %3 = tosa.rsqrt %2 : (tensor<6xi32>) -> tensor<6xi32>
  %4 = tosa.rsqrt %3 : (tensor<6xi32>) -> tensor<6xi32>
  %sharded = shard.shard %4 to %sharding annotate_for_users : tensor<6xi32>
  return %sharded : tensor<6xi32>
}

func.func @test_result_missing_sharding() -> tensor<6xi32> {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  // expected-error @+1 {{Cannot partition: user of result 0 must be shard.shard operation.}}
  %1 = tensor.empty() : tensor<6xi32>
  %3 = tosa.rsqrt %1 : (tensor<6xi32>) -> tensor<6xi32>
  %4 = shard.shard %3 to %sharding : tensor<6xi32>
  %sharded = shard.shard %4 to %sharding annotate_for_users : tensor<6xi32>
  return %sharded : tensor<6xi32>
}

func.func @test_multiple_users(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %1 = shard.shard %arg0 to %sharding : tensor<6xi32>
  %2 = shard.shard %1 to %sharding annotate_for_users : tensor<6xi32>
  // expected-error @+1 {{Cannot partition: result 0 must have exactly one use.}}
  %3 = tosa.rsqrt %2 : (tensor<6xi32>) -> tensor<6xi32>
  %4 = shard.shard %3 to %sharding : tensor<6xi32>
  %sharded = shard.shard %3 to %sharding annotate_for_users : tensor<6xi32>
  return %sharded : tensor<6xi32>
}

func.func @test_result_invalid_annotate(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %1 = shard.shard %arg0 to %sharding : tensor<6xi32>
  %2 = shard.shard %1 to %sharding annotate_for_users : tensor<6xi32>
  // expected-error @+1 {{Cannot partition: shard.shard for result 0 must not set 'annotate_for_users'.}}
  %3 = tosa.rsqrt %2 : (tensor<6xi32>) -> tensor<6xi32>
  %sharded = shard.shard %3 to %sharding annotate_for_users : tensor<6xi32>
  return %sharded : tensor<6xi32>
}
