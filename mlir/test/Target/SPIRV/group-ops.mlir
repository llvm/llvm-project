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
    // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : i32
    %0 = spirv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : i32
    spirv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @subgroup_block_read_intel_vector
  spirv.func @subgroup_block_read_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>) -> vector<3xi32> "None" {
    // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : vector<3xi32>
    %0 = spirv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : vector<3xi32>
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
}
