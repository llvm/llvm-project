// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @group_non_uniform_ballot
  spv.func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> "None" {
    // CHECK: %{{.*}} = spv.GroupNonUniformBallot "Workgroup" %{{.*}}: vector<4xi32>
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }

  // CHECK-LABEL: @group_non_uniform_broadcast
  spv.func @group_non_uniform_broadcast(%value: f32) -> f32 "None" {
    %one = spv.constant 1 : i32
    // CHECK: spv.GroupNonUniformBroadcast "Subgroup" %{{.*}}, %{{.*}} : f32, i32
    %0 = spv.GroupNonUniformBroadcast "Subgroup" %value, %one : f32, i32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_elect
  spv.func @group_non_uniform_elect() -> i1 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformElect "Workgroup" : i1
    %0 = spv.GroupNonUniformElect "Workgroup" : i1
    spv.ReturnValue %0: i1
  }

  // CHECK-LABEL: @group_non_uniform_fadd_reduce
  spv.func @group_non_uniform_fadd_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmax_reduce
  spv.func @group_non_uniform_fmax_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformFMax "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFMax "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmin_reduce
  spv.func @group_non_uniform_fmin_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformFMin "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFMin "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmul_reduce
  spv.func @group_non_uniform_fmul_reduce(%val: f32) -> f32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformFMul "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFMul "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_reduce
  spv.func @group_non_uniform_iadd_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_clustered_reduce
  spv.func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> "None" {
    %four = spv.constant 4 : i32
    // CHECK: %{{.+}} = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>
    %0 = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xi32>
    spv.ReturnValue %0: vector<2xi32>
  }

  // CHECK-LABEL: @group_non_uniform_imul_reduce
  spv.func @group_non_uniform_imul_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformIMul "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformIMul "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smax_reduce
  spv.func @group_non_uniform_smax_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformSMax "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformSMax "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smin_reduce
  spv.func @group_non_uniform_smin_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformSMin "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformSMin "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umax_reduce
  spv.func @group_non_uniform_umax_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformUMax "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformUMax "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umin_reduce
  spv.func @group_non_uniform_umin_reduce(%val: i32) -> i32 "None" {
    // CHECK: %{{.+}} = spv.GroupNonUniformUMin "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformUMin "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }
}
