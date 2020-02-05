// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @group_non_uniform_ballot
  func @group_non_uniform_ballot(%predicate: i1) -> vector<4xi32> {
    // CHECK: %{{.*}} = spv.GroupNonUniformBallot "Workgroup" %{{.*}}: vector<4xi32>
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }

  // CHECK-LABEL: @group_non_uniform_elect
  func @group_non_uniform_elect() -> i1 {
    // CHECK: %{{.+}} = spv.GroupNonUniformElect "Workgroup" : i1
    %0 = spv.GroupNonUniformElect "Workgroup" : i1
    spv.ReturnValue %0: i1
  }

  // CHECK-LABEL: @group_non_uniform_fadd_reduce
  func @group_non_uniform_fadd_reduce(%val: f32) -> f32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmax_reduce
  func @group_non_uniform_fmax_reduce(%val: f32) -> f32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformFMax "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFMax "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmin_reduce
  func @group_non_uniform_fmin_reduce(%val: f32) -> f32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformFMin "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFMin "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_fmul_reduce
  func @group_non_uniform_fmul_reduce(%val: f32) -> f32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformFMul "Workgroup" "Reduce" %{{.+}} : f32
    %0 = spv.GroupNonUniformFMul "Workgroup" "Reduce" %val : f32
    spv.ReturnValue %0: f32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_reduce
  func @group_non_uniform_iadd_reduce(%val: i32) -> i32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_iadd_clustered_reduce
  func @group_non_uniform_iadd_clustered_reduce(%val: vector<2xi32>) -> vector<2xi32> {
    %four = spv.constant 4 : i32
    // CHECK: %{{.+}} = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %{{.+}} cluster_size(%{{.+}}) : vector<2xi32>
    %0 = spv.GroupNonUniformIAdd "Workgroup" "ClusteredReduce" %val cluster_size(%four) : vector<2xi32>
    spv.ReturnValue %0: vector<2xi32>
  }

  // CHECK-LABEL: @group_non_uniform_imul_reduce
  func @group_non_uniform_imul_reduce(%val: i32) -> i32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformIMul "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformIMul "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smax_reduce
  func @group_non_uniform_smax_reduce(%val: i32) -> i32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformSMax "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformSMax "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_smin_reduce
  func @group_non_uniform_smin_reduce(%val: i32) -> i32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformSMin "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformSMin "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umax_reduce
  func @group_non_uniform_umax_reduce(%val: i32) -> i32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformUMax "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformUMax "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @group_non_uniform_umin_reduce
  func @group_non_uniform_umin_reduce(%val: i32) -> i32 {
    // CHECK: %{{.+}} = spv.GroupNonUniformUMin "Workgroup" "Reduce" %{{.+}} : i32
    %0 = spv.GroupNonUniformUMin "Workgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }
}
