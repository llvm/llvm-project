// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.SubgroupBallotKHR
//===----------------------------------------------------------------------===//

func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: %{{.*}} = spv.SubgroupBallotKHR %{{.*}} : vector<4xi32>
  %0 = spv.SubgroupBallotKHR %predicate: vector<4xi32>
  return %0: vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.GroupBroadcast
//===----------------------------------------------------------------------===//

func @group_broadcast_scalar(%value: f32, %localid: i32 ) -> f32 {
  // CHECK: spv.GroupBroadcast "Workgroup" %{{.*}}, %{{.*}} : f32, i32
  %0 = spv.GroupBroadcast "Workgroup" %value, %localid : f32, i32
  return %0: f32
}

// -----

func @group_broadcast_scalar_vector(%value: f32, %localid: vector<3xi32> ) -> f32 {
  // CHECK: spv.GroupBroadcast "Workgroup" %{{.*}}, %{{.*}} : f32, vector<3xi32>
  %0 = spv.GroupBroadcast "Workgroup" %value, %localid : f32, vector<3xi32>
  return %0: f32
}

// -----

func @group_broadcast_vector(%value: vector<4xf32>, %localid: vector<3xi32> ) -> vector<4xf32> {
  // CHECK: spv.GroupBroadcast "Subgroup" %{{.*}}, %{{.*}} : vector<4xf32>, vector<3xi32>
  %0 = spv.GroupBroadcast "Subgroup" %value, %localid : vector<4xf32>, vector<3xi32>
  return %0: vector<4xf32>
}

// -----

func @group_broadcast_negative_scope(%value: f32, %localid: vector<3xi32> ) -> f32 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}} 
  %0 = spv.GroupBroadcast "Device" %value, %localid : f32, vector<3xi32>
  return %0: f32
}

// -----

func @group_broadcast_negative_locid_dtype(%value: f32, %localid: vector<3xf32> ) -> f32 {
  // expected-error @+1 {{operand #1 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values}}
  %0 = spv.GroupBroadcast "Subgroup" %value, %localid : f32, vector<3xf32>
  return %0: f32
}

// -----

func @group_broadcast_negative_locid_vec4(%value: f32, %localid: vector<4xi32> ) -> f32 {
  // expected-error @+1 {{localid is a vector and can be with only  2 or 3 components, actual number is 4}}
  %0 = spv.GroupBroadcast "Subgroup" %value, %localid : f32, vector<4xi32>
  return %0: f32
}
