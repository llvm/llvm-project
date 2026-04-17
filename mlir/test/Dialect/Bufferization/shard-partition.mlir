// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(shard-partition))" \
// RUN:   %s | FileCheck %s

shard.grid @grid(shape = 4)

// CHECK-LABEL: func @test_alloc_tensor_op
// CHECK-SAME: tensor<?x2xf32>
func.func @test_alloc_tensor_op(%t: tensor<?x8xf32>, %sz: index)
{
  %sharding = shard.sharding @grid split_axes = [[], [0]] : !shard.sharding
  %sharded = shard.shard %t to %sharding : tensor<?x8xf32>
  // CHECK: bufferization.alloc_tensor(%{{.*}}) : tensor<?x2xf32>
  %0 = bufferization.alloc_tensor(%sz) : tensor<?x8xf32>
  %sharded0 = shard.shard %0 to %sharding : tensor<?x8xf32>
  %sharded1 = shard.shard %sharded to %sharding annotate_for_users : tensor<?x8xf32>
  // CHECK: bufferization.alloc_tensor() copy(%{{.*}}) {escape = true} : tensor<?x2xf32>
  %4 = bufferization.alloc_tensor() copy(%sharded1) {escape = true} : tensor<?x8xf32>
  %sharded4 = shard.shard %4 to %sharding : tensor<?x8xf32>
  return
}

// CHECK-LABEL: func @test_dealloc_tensor_op
// CHECK-SAME: tensor<1xi32>
func.func @test_dealloc_tensor_op(%arg0: tensor<4xi32>) {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %sharded = shard.shard %arg0 to %sharding : tensor<4xi32>
  %sharded1 = shard.shard %sharded to %sharding annotate_for_users : tensor<4xi32>
  // CHECK: bufferization.dealloc_tensor {{.*}} : tensor<1xi32>
  bufferization.dealloc_tensor %sharded1 : tensor<4xi32>
  return
}

// CHECK-LABEL: func @test_materialize_in_destination_op
// CHECK-SAME: tensor<2xf32>) -> tensor<2xf32>
func.func @test_materialize_in_destination_op(
    %arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %sharding = shard.sharding @grid split_axes = [[0]] : !shard.sharding
  %sharded0 = shard.shard %arg0 to %sharding : tensor<?xf32>
  %sharded1 = shard.shard %arg1 to %sharding : tensor<?xf32>
  %sharded2 = shard.shard %arg2 to %sharding : tensor<8xf32>
  %sharded0_in = shard.shard %sharded0 to %sharding annotate_for_users : tensor<?xf32>
  %sharded1_in = shard.shard %sharded1 to %sharding annotate_for_users : tensor<?xf32>
  // CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = bufferization.materialize_in_destination %sharded0_in in %sharded1_in : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %sharded_res0 = shard.shard %0 to %sharding : tensor<?xf32>
  // CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<?xf32>, tensor<2xf32>) -> tensor<2xf32>
  %sharded0_in2 = shard.shard %sharded0 to %sharding annotate_for_users : tensor<?xf32>
  %sharded2_in1 = shard.shard %sharded2 to %sharding annotate_for_users : tensor<8xf32>
  %1 = bufferization.materialize_in_destination %sharded0_in2 in %sharded2_in1 : (tensor<?xf32>, tensor<8xf32>) -> tensor<8xf32>
  %sharded_res1 = shard.shard %1 to %sharding : tensor<8xf32>
  %sharded_res1_in = shard.shard %sharded_res1 to %sharding annotate_for_users : tensor<8xf32>
  // CHECK tensor<2xf32>
  return %sharded_res1_in : tensor<8xf32>
}
