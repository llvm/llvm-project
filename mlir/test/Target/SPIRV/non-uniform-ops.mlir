// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader, Linkage, GroupNonUniformBallot, GroupNonUniformArithmetic, GroupNonUniformClustered, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformVote, GroupNonUniformQuad], []> {
  // CHECK-LABEL: @group_non_uniform_ballot
  spirv.func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> "None" {
    // CHECK: %{{.*}} = spirv.GroupNonUniformBallot <Subgroup> %{{.*}}: vector<4xi32>
  %0 = spirv.GroupNonUniformBallot <Subgroup> %predicate : vector<4xi32>
    spirv.ReturnValue %0: vector<4xi32>
  }

  // CHECK-LABEL: @group_non_uniform_broadcast
  spirv.func @group_non_uniform_broadcast(%value: f32) -> f32 "None" {
    %one = spirv.Constant 1 : i32
    // CHECK: spirv.GroupNonUniformBroadcast <Subgroup> %{{.*}}, %{{.*}} : f32, i32
    %0 = spirv.GroupNonUniformBroadcast <Subgroup> %value, %one : f32, i32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_broadcast_first
  spirv.func @group_non_uniform_broadcast_first(%value: f32) -> f32 "None" {
    // CHECK: spirv.GroupNonUniformBroadcastFirst <Subgroup> %{{.*}} : f32
    %0 = spirv.GroupNonUniformBroadcastFirst <Subgroup> %value : f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_elect
  spirv.func @group_non_uniform_elect() -> i1 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformElect <Subgroup> : i1
    %0 = spirv.GroupNonUniformElect <Subgroup> : i1
    spirv.ReturnValue %0: i1
  }

  // CHECK-LABEL: @group_non_uniform_fadd_reduce
  spirv.func @group_non_uniform_fadd_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFAdd <Subgroup> <Reduce> %{{.+}} : f32 -> f32
    %0 = spirv.GroupNonUniformFAdd <Subgroup> <Reduce> %val : f32 -> f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmax_reduce
  spirv.func @group_non_uniform_fmax_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFMax <Subgroup> <Reduce> %{{.+}} : f32 -> f32
    %0 = spirv.GroupNonUniformFMax <Subgroup> <Reduce> %val : f32 -> f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmin_reduce
  spirv.func @group_non_uniform_fmin_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFMin <Subgroup> <Reduce> %{{.+}} : f32 -> f32
    %0 = spirv.GroupNonUniformFMin <Subgroup> <Reduce> %val : f32 -> f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmul_reduce
  spirv.func @group_non_uniform_fmul_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFMul <Subgroup> <Reduce> %{{.+}} : f32 -> f32
    %0 = spirv.GroupNonUniformFMul <Subgroup> <Reduce> %val : f32 -> f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_reduce
  spirv.func @group_non_uniform_iadd_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformIAdd <Subgroup> <Reduce> %{{.+}} : i32 -> i32
    %0 = spirv.GroupNonUniformIAdd <Subgroup> <Reduce> %val : i32 -> i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_clustered_reduce
  spirv.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> "None" {
    %four = spirv.Constant 4 : i32
    // CHECK: %{{.+}} = spirv.GroupNonUniformIAdd <Subgroup> <ClusteredReduce> %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>, i32 -> vector<2xi32>
    %0 = spirv.GroupNonUniformIAdd <Subgroup> <ClusteredReduce> %val cluster_size(%four) : vector<2xi32>, i32 -> vector<2xi32>
    spirv.ReturnValue %0: vector<2xi32>
  }

  // CHECK-LABEL: @group_non_uniform_imul_reduce
  spirv.func @group_non_uniform_imul_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformIMul <Subgroup> <Reduce> %{{.+}} : i32 -> i32
    %0 = spirv.GroupNonUniformIMul <Subgroup> <Reduce> %val : i32 -> i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smax_reduce
  spirv.func @group_non_uniform_smax_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformSMax <Subgroup> <Reduce> %{{.+}} : i32 -> i32
    %0 = spirv.GroupNonUniformSMax <Subgroup> <Reduce> %val : i32 -> i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smin_reduce
  spirv.func @group_non_uniform_smin_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformSMin <Subgroup> <Reduce> %{{.+}} : i32 -> i32
    %0 = spirv.GroupNonUniformSMin <Subgroup> <Reduce> %val : i32 -> i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umax_reduce
  spirv.func @group_non_uniform_umax_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformUMax <Subgroup> <Reduce> %{{.+}} : i32 -> i32
    %0 = spirv.GroupNonUniformUMax <Subgroup> <Reduce> %val : i32 -> i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umin_reduce
  spirv.func @group_non_uniform_umin_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformUMin <Subgroup> <Reduce> %{{.+}} : i32 -> i32
    %0 = spirv.GroupNonUniformUMin <Subgroup> <Reduce> %val : i32 -> i32
    spirv.ReturnValue %0: i32
  }

  spirv.func @group_non_uniform_shuffle(%val: f32, %id: i32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffle <Subgroup> %{{.+}}, %{{.+}} : f32, i32
    %0 = spirv.GroupNonUniformShuffle <Subgroup> %val, %id : f32, i32
    spirv.ReturnValue %0: f32
  }

  spirv.func @group_non_uniform_shuffle_up(%val: f32, %id: i32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleUp <Subgroup> %{{.+}}, %{{.+}} : f32, i32
    %0 = spirv.GroupNonUniformShuffleUp <Subgroup> %val, %id : f32, i32
    spirv.ReturnValue %0: f32
  }

  spirv.func @group_non_uniform_shuffle_down(%val: f32, %id: i32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleDown <Subgroup> %{{.+}}, %{{.+}} : f32, i32
    %0 = spirv.GroupNonUniformShuffleDown <Subgroup> %val, %id : f32, i32
    spirv.ReturnValue %0: f32
  }

  spirv.func @group_non_uniform_shuffle_xor(%val: f32, %id: i32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleXor <Subgroup> %{{.+}}, %{{.+}} : f32, i32
    %0 = spirv.GroupNonUniformShuffleXor <Subgroup> %val, %id : f32, i32
    spirv.ReturnValue %0: f32
  }

  spirv.func @group_non_uniform_all(%pred: i1) -> i1 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformAll <Subgroup> %{{.+}} : i1
    %0 = spirv.GroupNonUniformAll <Subgroup> %pred : i1
    spirv.ReturnValue %0: i1
  }

  spirv.func @group_non_uniform_any(%pred: i1) -> i1 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformAny <Subgroup> %{{.+}} : i1
    %0 = spirv.GroupNonUniformAny <Subgroup> %pred : i1
    spirv.ReturnValue %0: i1
  }

  spirv.func @group_non_uniform_all_equal(%val: vector<4xi32>) -> i1 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformAllEqual <Subgroup> %{{.+}} : vector<4xi32>, i1
    %0 = spirv.GroupNonUniformAllEqual <Subgroup> %val : vector<4xi32>, i1
    spirv.ReturnValue %0: i1
  }

  spirv.func @group_non_uniform_quad_swap_vec(%val: vector<4xf32>) -> vector<4xf32> "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformQuadSwap <Subgroup> <Vertical> %{{.+}} : vector<4xf32>
    %0 = spirv.GroupNonUniformQuadSwap <Subgroup> <Vertical> %val : vector<4xf32>
    spirv.ReturnValue %0: vector<4xf32>
  }

  spirv.func @group_non_uniform_quad_swap_scalar(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformQuadSwap <Subgroup> <Horizontal> %{{.+}} : f32
    %0 = spirv.GroupNonUniformQuadSwap <Subgroup> <Horizontal> %val : f32
    spirv.ReturnValue %0: f32
  }
}
