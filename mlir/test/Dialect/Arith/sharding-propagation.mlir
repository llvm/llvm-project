// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation))" %s | FileCheck %s

mesh.mesh @mesh4x4(shape = 4x4)

// CHECK-LABEL: func.func @test_shard_constant() -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
// CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated:%.*]] = mesh.shard [[vcst]] to [[vsharding]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vcst_0:%.*]] = arith.constant 4.340000e+01 : f32
// CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_1:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_2:%.*]] = mesh.shard [[v0]] to [[vsharding_1]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_3:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_4:%.*]] = mesh.shard [[vsharding_annotated]] to [[vsharding_3]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_5:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_6:%.*]] = mesh.shard [[vsharding_annotated_2]] to [[vsharding_5]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[v1:%.*]] = linalg.add ins([[vsharding_annotated_4]], [[vcst_0]] : tensor<1024x1024xf32>, f32) outs([[vsharding_annotated_6]] : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_7:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_8:%.*]] = mesh.shard [[v1]] to [[vsharding_7]] : tensor<1024x1024xf32>
// CHECK-NEXT: return [[vsharding_annotated_8]] : tensor<1024x1024xf32>
func.func @test_shard_constant() -> (tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %sharding_1 = mesh.sharding @mesh4x4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_1 = mesh.shard %cst_1 to %sharding_1 : tensor<1024x1024xf32>
    %ci = arith.constant 43.4e+00 : f32
    %o1 = tensor.empty() : tensor<1024x1024xf32>
    %res = linalg.add ins(%sharding_annotated_1, %ci : tensor<1024x1024xf32>, f32) outs(%o1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %res : tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @test_shard_constant_back() -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
// CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated:%.*]] = mesh.shard [[vcst]] to [[vsharding]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vcst_0:%.*]] = arith.constant 4.340000e+01 : f32
// CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_1:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_2:%.*]] = mesh.shard [[v0]] to [[vsharding_1]] : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_3:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_4:%.*]] = mesh.shard [[vsharding_annotated]] to [[vsharding_3]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_5:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_6:%.*]] = mesh.shard [[vsharding_annotated_2]] to [[vsharding_5]] annotate_for_users : tensor<1024x1024xf32>
// CHECK-NEXT: [[v1:%.*]] = linalg.add ins([[vsharding_annotated_4]], [[vcst_0]] : tensor<1024x1024xf32>, f32) outs([[vsharding_annotated_6]] : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK-NEXT: [[vsharding_7:%.*]] = mesh.sharding @mesh4x4 split_axes = {{\[\[}}0]] : !mesh.sharding
// CHECK-NEXT: [[vsharding_annotated_8:%.*]] = mesh.shard [[v1]] to [[vsharding_7]] : tensor<1024x1024xf32>
func.func @test_shard_constant_back() -> (tensor<1024x1024xf32>) attributes {llvm.emit_c_interface} {
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024x1024xf32>
    %ci = arith.constant 43.4e+00 : f32
    %o1 = tensor.empty() : tensor<1024x1024xf32>
    %res = linalg.add ins(%cst_1, %ci : tensor<1024x1024xf32>, f32) outs(%o1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %sharding_1 = mesh.sharding @mesh4x4 split_axes = [[0]] : !mesh.sharding
    %sharding_annotated_1 = mesh.shard %res to %sharding_1 : tensor<1024x1024xf32>
    return %sharding_annotated_1 : tensor<1024x1024xf32>
}
