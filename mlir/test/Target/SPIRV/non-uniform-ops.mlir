// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @group_non_uniform_ballot
  spirv.func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> "None" {
    // CHECK: %{{.*}} = spirv.GroupNonUniformBallot <Workgroup> %{{.*}}: vector<4xi32>
  %0 = spirv.GroupNonUniformBallot <Workgroup> %predicate : vector<4xi32>
    spirv.ReturnValue %0: vector<4xi32>
  }

  // CHECK-LABEL: @group_non_uniform_broadcast
  spirv.func @group_non_uniform_broadcast(%value: f32) -> f32 "None" {
    %one = spirv.Constant 1 : i32
    // CHECK: spirv.GroupNonUniformBroadcast <Subgroup> %{{.*}}, %{{.*}} : f32, i32
    %0 = spirv.GroupNonUniformBroadcast <Subgroup> %value, %one : f32, i32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_elect
  spirv.func @group_non_uniform_elect() -> i1 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformElect <Workgroup> : i1
    %0 = spirv.GroupNonUniformElect <Workgroup> : i1
    spirv.ReturnValue %0: i1
  }

  // CHECK-LABEL: @group_non_uniform_fadd_reduce
  spirv.func @group_non_uniform_fadd_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFAdd "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spirv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmax_reduce
  spirv.func @group_non_uniform_fmax_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFMax "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spirv.GroupNonUniformFMax "Workgroup" "Reduce" %val : f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmin_reduce
  spirv.func @group_non_uniform_fmin_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFMin "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spirv.GroupNonUniformFMin "Workgroup" "Reduce" %val : f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmul_reduce
  spirv.func @group_non_uniform_fmul_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformFMul "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spirv.GroupNonUniformFMul "Workgroup" "Reduce" %val : f32
    spirv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_reduce
  spirv.func @group_non_uniform_iadd_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformIAdd "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spirv.GroupNonUniformIAdd "Workgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_clustered_reduce
  spirv.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> "None" {
    %four = spirv.Constant 4 : i32
    // CHECK: %{{.+}} = spirv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>
    %0 = spirv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xi32>
    spirv.ReturnValue %0: vector<2xi32>
  }

  // CHECK-LABEL: @group_non_uniform_imul_reduce
  spirv.func @group_non_uniform_imul_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformIMul "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spirv.GroupNonUniformIMul "Workgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smax_reduce
  spirv.func @group_non_uniform_smax_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformSMax "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spirv.GroupNonUniformSMax "Workgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smin_reduce
  spirv.func @group_non_uniform_smin_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformSMin "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spirv.GroupNonUniformSMin "Workgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umax_reduce
  spirv.func @group_non_uniform_umax_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformUMax "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spirv.GroupNonUniformUMax "Workgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umin_reduce
  spirv.func @group_non_uniform_umin_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spirv.GroupNonUniformUMin "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spirv.GroupNonUniformUMin "Workgroup" "Reduce" %val : i32
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
}
