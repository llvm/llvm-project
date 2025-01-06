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
  // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : !spirv.ptr<i32, StorageBuffer> -> i32
  %0 = spirv.INTEL.SubgroupBlockRead %ptr : !spirv.ptr<i32, StorageBuffer> -> i32
  return %0: i32
}

// -----

func.func @subgroup_block_read_intel_vector(%ptr : !spirv.ptr<i32, StorageBuffer>) -> vector<3xi32> {
  // CHECK: spirv.INTEL.SubgroupBlockRead %{{.*}} : !spirv.ptr<i32, StorageBuffer> -> vector<3xi32>
  %0 = spirv.INTEL.SubgroupBlockRead %ptr : !spirv.ptr<i32, StorageBuffer> -> vector<3xi32>
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

// -----

//===----------------------------------------------------------------------===//
// Group ops
//===----------------------------------------------------------------------===//

func.func @group_iadd(%value: i32) -> i32 {
  // CHECK: spirv.GroupIAdd <Workgroup> <Reduce> %{{.*}} : i32
  %0 = spirv.GroupIAdd <Workgroup> <Reduce> %value : i32
  return %0: i32
}

// -----

func.func @group_fadd(%value: f32) -> f32 {
  // CHECK: spirv.GroupFAdd <Workgroup> <Reduce> %{{.*}} : f32
  %0 = spirv.GroupFAdd <Workgroup> <Reduce> %value : f32
  return %0: f32
}

// -----

func.func @group_fmin(%value: f32) -> f32 {
  // CHECK: spirv.GroupFMin <Workgroup> <Reduce> %{{.*}} : f32
  %0 = spirv.GroupFMin <Workgroup> <Reduce> %value : f32
  return %0: f32
}

// -----

func.func @group_umin(%value: i32) -> i32 {
  // CHECK: spirv.GroupUMin <Workgroup> <Reduce> %{{.*}} : i32
  %0 = spirv.GroupUMin <Workgroup> <Reduce> %value : i32
  return %0: i32
}

// -----

func.func @group_smin(%value: i32) -> i32 {
  // CHECK: spirv.GroupSMin <Workgroup> <Reduce> %{{.*}} : i32
  %0 = spirv.GroupSMin <Workgroup> <Reduce> %value : i32
  return %0: i32
}

// -----

func.func @group_fmax(%value: f32) -> f32 {
  // CHECK: spirv.GroupFMax <Workgroup> <Reduce> %{{.*}} : f32
  %0 = spirv.GroupFMax <Workgroup> <Reduce> %value : f32
  return %0: f32
}

// -----

func.func @group_umax(%value: i32) -> i32 {
  // CHECK: spirv.GroupUMax <Workgroup> <Reduce> %{{.*}} : i32
  %0 = spirv.GroupUMax <Workgroup> <Reduce> %value : i32
  return %0: i32
}

// -----

func.func @group_smax(%value: i32) -> i32 {
  // CHECK: spirv.GroupSMax <Workgroup> <Reduce> %{{.*}} : i32
  %0 = spirv.GroupSMax <Workgroup> <Reduce> %value : i32
  return %0: i32
}

// -----

func.func @group_imul(%value: i32) -> i32 {
  // CHECK: spirv.KHR.GroupIMul <Workgroup> <Reduce> %{{.*}} : i32
  %0 = spirv.KHR.GroupIMul <Workgroup> <Reduce> %value : i32
  return %0: i32
}

// -----

func.func @group_fmul(%value: f32) -> f32 {
  // CHECK: spirv.KHR.GroupFMul <Workgroup> <Reduce> %{{.*}} : f32
  %0 = spirv.KHR.GroupFMul <Workgroup> <Reduce> %value : f32
  return %0: f32
}
