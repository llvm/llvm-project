// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation))" %s | FileCheck %s

shard.grid @grid4x4(shape = 4x4)

// CHECK-LABEL: func.func @test_shard_constant() -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
// CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded:%.*]] = shard.shard [[vcst]] to [[vsharding]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vcst_0:%.*]] = arith.constant 4.340000e+01 : f32
// CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_1:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_2:%.*]] = shard.shard [[v0]] to [[vsharding_1]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_3:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_4:%.*]] = shard.shard [[vsharded]] to [[vsharding_3]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_5:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_6:%.*]] = shard.shard [[vsharded_2]] to [[vsharding_5]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[v1:%.*]] = linalg.add ins([[vsharded_4]], [[vcst_0]] : tensor<1024x1024xf32>, f32) outs([[vsharded_6]] : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_7:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_8:%.*]] = shard.shard [[v1]] to [[vsharding_7]] : tensor<1024x1024xf32>
// CHECK-NEXT: return [[vsharded_8]] : tensor<1024x1024xf32>
func.func @test_shard_constant() -> (tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %sharding_1 = shard.sharding @grid4x4 split_axes = [[0]] : !shard.sharding
    %sharded_1 = shard.shard %cst_1 to %sharding_1 : tensor<1024x1024xf32>
    %ci = arith.constant 43.4e+00 : f32
    %o1 = tensor.empty() : tensor<1024x1024xf32>
    %res = linalg.add ins(%sharded_1, %ci : tensor<1024x1024xf32>, f32) outs(%o1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %res : tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @test_shard_constant_back() -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
// CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded:%.*]] = shard.shard [[vcst]] to [[vsharding]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vcst_0:%.*]] = arith.constant 4.340000e+01 : f32
// CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_1:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_2:%.*]] = shard.shard [[v0]] to [[vsharding_1]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_3:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_4:%.*]] = shard.shard [[vsharded]] to [[vsharding_3]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_5:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_6:%.*]] = shard.shard [[vsharded_2]] to [[vsharding_5]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[v1:%.*]] = linalg.add ins([[vsharded_4]], [[vcst_0]] : tensor<1024x1024xf32>, f32) outs([[vsharded_6]] : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_7:%.*]] = shard.sharding @grid4x4 split_axes = {{\[\[}}0]] : !shard.sharding
// CHECK-NEXT: [[vsharded_8:%.*]] = shard.shard [[v1]] to [[vsharding_7]] : tensor<1024x1024xf32>
func.func @test_shard_constant_back() -> (tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %ci = arith.constant 43.4e+00 : f32
    %o1 = tensor.empty() : tensor<1024x1024xf32>
    %res = linalg.add ins(%cst_1, %ci : tensor<1024x1024xf32>, f32) outs(%o1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %sharding_1 = shard.sharding @grid4x4 split_axes = [[0]] : !shard.sharding
    %sharded_1 = shard.shard %res to %sharding_1 : tensor<1024x1024xf32>
    return %sharded_1 : tensor<1024x1024xf32>
}
