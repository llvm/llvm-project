// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @subgroup_ballot
  spv.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> "None" {
    // CHECK: %{{.*}} = spv.SubgroupBallotKHR %{{.*}}: vector<4xi32>
    %0 = spv.SubgroupBallotKHR %predicate: vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
  // CHECK-LABEL: @group_broadcast_1
  spv.func @group_broadcast_1(%value: f32, %localid: i32 ) -> f32 "None" {
    // CHECK: spv.GroupBroadcast "Workgroup" %{{.*}}, %{{.*}} : f32, i32
    %0 = spv.GroupBroadcast "Workgroup" %value, %localid : f32, i32
    spv.ReturnValue %0: f32
  }
  // CHECK-LABEL: @group_broadcast_2
  spv.func @group_broadcast_2(%value: f32, %localid: vector<3xi32> ) -> f32 "None" {
    // CHECK: spv.GroupBroadcast "Workgroup" %{{.*}}, %{{.*}} : f32, vector<3xi32>
    %0 = spv.GroupBroadcast "Workgroup" %value, %localid : f32, vector<3xi32>
    spv.ReturnValue %0: f32
  }
}
