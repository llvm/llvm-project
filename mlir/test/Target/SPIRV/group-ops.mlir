// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
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
  // CHECK-LABEL: @subgroup_block_read_intel
  spirv.func @subgroup_block_read_intel(%ptr : !spirv.ptr<i32, StorageBuffer>) -> i32 "None" {
    // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : !spirv.ptr<i32, StorageBuffer> -> i32
    %0 = spirv.INTEL.SubgroupBlockRead %ptr : !spirv.ptr<i32, StorageBuffer> -> i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @subgroup_block_read_intel_vector
  spirv.func @subgroup_block_read_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>) -> vector<3xi32> "None" {
    // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : !spirv.ptr<i32, StorageBuffer> -> vector<3xi32>
    %0 = spirv.INTEL.SubgroupBlockRead %ptr : !spirv.ptr<i32, StorageBuffer> -> vector<3xi32>
    spirv.ReturnValue %0: vector<3xi32>
  }
  // CHECK-LABEL: @subgroup_block_write_intel
  spirv.func @subgroup_block_write_intel(%ptr : !spirv.ptr<i32, StorageBuffer>, %value: i32) -> () "None" {
    // CHECK: spirv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : i32
    spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : i32
    spirv.Return
  }
  // CHECK-LABEL: @subgroup_block_write_intel_vector
  spirv.func @subgroup_block_write_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>, %value: vector<3xi32>) -> () "None" {
    // CHECK: spirv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : vector<3xi32>
    spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : vector<3xi32>
    spirv.Return
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
