// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @subgroup_ballot
  spv.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> "None" {
    // CHECK: %{{.*}} = spv.KHR.SubgroupBallot %{{.*}}: vector<4xi32>
    %0 = spv.KHR.SubgroupBallot %predicate: vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
  // CHECK-LABEL: @group_broadcast_1
  spv.func @group_broadcast_1(%value: f32, %localid: i32 ) -> f32 "None" {
    // CHECK: spv.GroupBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, i32
    %0 = spv.GroupBroadcast <Workgroup> %value, %localid : f32, i32
    spv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_broadcast_2
  spv.func @group_broadcast_2(%value: f32, %localid: vector<3xi32> ) -> f32 "None" {
    // CHECK: spv.GroupBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, vector<3xi32>
    %0 = spv.GroupBroadcast <Workgroup> %value, %localid : f32, vector<3xi32>
    spv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @subgroup_block_read_intel
  spv.func @subgroup_block_read_intel(%ptr : !spv.ptr<i32, StorageBuffer>) -> i32 "None" {
    // CHECK: spv.INTEL.SubgroupBlockRead %{{.*}} : i32
    %0 = spv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : i32
    spv.ReturnValue %0: i32
  }
  // CHECK-LABEL: @subgroup_block_read_intel_vector
  spv.func @subgroup_block_read_intel_vector(%ptr : !spv.ptr<i32, StorageBuffer>) -> vector<3xi32> "None" {
    // CHECK: spv.INTEL.SubgroupBlockRead %{{.*}} : vector<3xi32>
    %0 = spv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : vector<3xi32>
    spv.ReturnValue %0: vector<3xi32>
  }
  // CHECK-LABEL: @subgroup_block_write_intel
  spv.func @subgroup_block_write_intel(%ptr : !spv.ptr<i32, StorageBuffer>, %value: i32) -> () "None" {
    // CHECK: spv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : i32
    spv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : i32
    spv.Return
  }
  // CHECK-LABEL: @subgroup_block_write_intel_vector
  spv.func @subgroup_block_write_intel_vector(%ptr : !spv.ptr<i32, StorageBuffer>, %value: vector<3xi32>) -> () "None" {
    // CHECK: spv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : vector<3xi32>
    spv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : vector<3xi32>
    spv.Return
  }
}
