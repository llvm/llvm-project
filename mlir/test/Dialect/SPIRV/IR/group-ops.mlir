// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.KHR.SubgroupBallot
//===----------------------------------------------------------------------===//

func.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: %{{.*}} = spirv.KHR.SubgroupBallot %{{.*}} : vector<4xi32>
  %0 = spirv.KHR.SubgroupBallot %predicate: vector<4xi32>
  return %0: vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GroupBroadcast
//===----------------------------------------------------------------------===//

func.func @group_broadcast_scalar(%value: f32, %localid: i32 ) -> f32 {
  // CHECK: spirv.GroupBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, i32
  %0 = spirv.GroupBroadcast <Workgroup> %value, %localid : f32, i32
  return %0: f32
}

// -----

func.func @group_broadcast_scalar_vector(%value: f32, %localid: vector<3xi32> ) -> f32 {
  // CHECK: spirv.GroupBroadcast <Workgroup> %{{.*}}, %{{.*}} : f32, vector<3xi32>
  %0 = spirv.GroupBroadcast <Workgroup> %value, %localid : f32, vector<3xi32>
  return %0: f32
}

// -----

func.func @group_broadcast_vector(%value: vector<4xf32>, %localid: vector<3xi32> ) -> vector<4xf32> {
  // CHECK: spirv.GroupBroadcast <Subgroup> %{{.*}}, %{{.*}} : vector<4xf32>, vector<3xi32>
  %0 = spirv.GroupBroadcast <Subgroup> %value, %localid : vector<4xf32>, vector<3xi32>
  return %0: vector<4xf32>
}

// -----

func.func @group_broadcast_negative_scope(%value: f32, %localid: vector<3xi32> ) -> f32 {
  // expected-error @+1 {{execution scope must be 'Workgroup' or 'Subgroup'}} 
  %0 = spirv.GroupBroadcast <Device> %value, %localid : f32, vector<3xi32>
  return %0: f32
}

// -----

func.func @group_broadcast_negative_locid_dtype(%value: f32, %localid: vector<3xf32> ) -> f32 {
  // expected-error @+1 {{operand #1 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values}}
  %0 = spirv.GroupBroadcast <Subgroup> %value, %localid : f32, vector<3xf32>
  return %0: f32
}

// -----

func.func @group_broadcast_negative_locid_vec4(%value: f32, %localid: vector<4xi32> ) -> f32 {
  // expected-error @+1 {{localid is a vector and can be with only  2 or 3 components, actual number is 4}}
  %0 = spirv.GroupBroadcast <Subgroup> %value, %localid : f32, vector<4xi32>
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.KHR.SubgroupBallot
//===----------------------------------------------------------------------===//

func.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  %0 = spirv.KHR.SubgroupBallot %predicate: vector<4xi32>
  return %0: vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.SubgroupBlockRead
//===----------------------------------------------------------------------===//

func.func @subgroup_block_read_intel(%ptr : !spirv.ptr<i32, StorageBuffer>) -> i32 {
  // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : i32
  %0 = spirv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : i32
  return %0: i32
}

// -----

func.func @subgroup_block_read_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>) -> vector<3xi32> {
  // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : vector<3xi32>
  %0 = spirv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : vector<3xi32>
  return %0: vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INTEL.SubgroupBlockWrite
//===----------------------------------------------------------------------===//

func.func @subgroup_block_write_intel(%ptr : !spirv.ptr<i32, StorageBuffer>, %value: i32) -> () {
  // CHECK: spirv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : i32
  spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : i32
  return
}

// -----

func.func @subgroup_block_write_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>, %value: vector<3xi32>) -> () {
  // CHECK: spirv.INTEL.SubgroupBlockWrite %{{.*}}, %{{.*}} : vector<3xi32>
  spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : vector<3xi32>
  return
}
