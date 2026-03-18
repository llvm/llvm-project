// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip --split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.3,
  [Shader, Linkage, SubgroupBallotKHR, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR],
  [SPV_KHR_storage_buffer_storage_class, SPV_KHR_shader_ballot, SPV_KHR_uniform_group_instructions]> {
  // CHECK-LABEL: @subgroup_ballot
  spirv.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> "None" {
    // CHECK: %{{.*}} = spirv.KHR.SubgroupBallot %{{.*}}: vector<4xi32>
    %0 = spirv.KHR.SubgroupBallot %predicate: vector<4xi32>
    spirv.ReturnValue %0: vector<4xi32>
  }
  // CHECK-LABEL: @group_broadcast_1
  spirv.func @group_broadcast_1(%value: f32, %localid: i32 ) -> f32 "None" {
    // CHECK: spirv.GroupBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, i32
    %0 = spirv.GroupBroadcast <Workgroup> %value, %localid : f32, i32
    spirv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_broadcast_2
  spirv.func @group_broadcast_2(%value: f32, %localid: vector<3xi32> ) -> f32 "None" {
    // CHECK: spirv.GroupBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, vector<3xi32>
    %0 = spirv.GroupBroadcast <Workgroup> %value, %localid : f32, vector<3xi32>
    spirv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_iadd
  spirv.func @group_iadd(%value: i32) -> i32 "None" {
    // CHECK: spirv.GroupIAdd <Workgroup> <Reduce> %{{.*}} : i32
    %0 = spirv.GroupIAdd <Workgroup> <Reduce> %value : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @group_fadd
  spirv.func @group_fadd(%value: f32) -> f32 "None" {
    // CHECK: spirv.GroupFAdd <Workgroup> <Reduce> %{{.*}} : f32
    %0 = spirv.GroupFAdd <Workgroup> <Reduce> %value : f32
    spirv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_fmin
  spirv.func @group_fmin(%value: f32) -> f32 "None" {
    // CHECK: spirv.GroupFMin <Workgroup> <Reduce> %{{.*}} : f32
    %0 = spirv.GroupFMin <Workgroup> <Reduce> %value : f32
    spirv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_umin
  spirv.func @group_umin(%value: i32) -> i32 "None" {
    // CHECK: spirv.GroupUMin <Workgroup> <Reduce> %{{.*}} : i32
    %0 = spirv.GroupUMin <Workgroup> <Reduce> %value : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @group_smin
  spirv.func @group_smin(%value: i32) -> i32 "None" {
    // CHECK: spirv.GroupSMin <Workgroup> <Reduce> %{{.*}} : i32
    %0 = spirv.GroupSMin <Workgroup> <Reduce> %value : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @group_fmax
  spirv.func @group_fmax(%value: f32) -> f32 "None" {
    // CHECK: spirv.GroupFMax <Workgroup> <Reduce> %{{.*}} : f32
    %0 = spirv.GroupFMax <Workgroup> <Reduce> %value : f32
    spirv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_umax
  spirv.func @group_umax(%value: i32) -> i32 "None" {
    // CHECK: spirv.GroupUMax <Workgroup> <Reduce> %{{.*}} : i32
    %0 = spirv.GroupUMax <Workgroup> <Reduce> %value : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @group_smax
  spirv.func @group_smax(%value: i32) -> i32 "None" {
    // CHECK: spirv.GroupSMax <Workgroup> <Reduce> %{{.*}} : i32
    %0 = spirv.GroupSMax <Workgroup> <Reduce> %value : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @group_imul
  spirv.func @group_imul(%value: i32) -> i32 "None" {
    // CHECK: spirv.KHR.GroupIMul <Workgroup> <Reduce> %{{.*}} : i32
    %0 = spirv.KHR.GroupIMul <Workgroup> <Reduce> %value : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @group_fmul
  spirv.func @group_fmul(%value: f32) -> f32 "None" {
    // CHECK: spirv.KHR.GroupFMul <Workgroup> <Reduce> %{{.*}} : f32
    %0 = spirv.KHR.GroupFMul <Workgroup> <Reduce> %value : f32
    spirv.ReturnValue %0: f32
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader, GroupNonUniformBallot, Linkage], []> {
  // CHECK-LABEL: @group_non_uniform_ballot_bit_count
  spirv.func @group_non_uniform_ballot_bit_count(%value: vector<4xi32>) -> i32 "None" {
    // CHECK: spirv.GroupNonUniformBallotBitCount <Subgroup> <Reduce> {{%.*}} : vector<4xi32> -> i32
    %0 = spirv.GroupNonUniformBallotBitCount <Subgroup> <Reduce> %value : vector<4xi32> -> i32
    spirv.ReturnValue %0 : i32
  }
}
