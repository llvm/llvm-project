// RUN: mlir-opt -mlir-disable-threading -test-spirv-op-availability %s | FileCheck %s

// CHECK-LABEL: iadd
func.func @iadd(%arg: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.5
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ ]
  %0 = spv.IAdd %arg, %arg: i32
  return %0: i32
}

// CHECK: atomic_compare_exchange_weak
func.func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.3
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [Kernel] ]
  %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
  return %0: i32
}

// CHECK-LABEL: subgroup_ballot
func.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: min version: v1.3
  // CHECK: max version: v1.5
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [GroupNonUniformBallot] ]
  %0 = spv.GroupNonUniformBallot Workgroup %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// CHECK-LABEL: module_logical_glsl450
func.func @module_logical_glsl450() {
  // CHECK: spv.module min version: v1.0
  // CHECK: spv.module max version: v1.5
  // CHECK: spv.module extensions: [ ]
  // CHECK: spv.module capabilities: [ [Shader] ]
  spv.module Logical GLSL450 { }
  return
}

// CHECK-LABEL: module_physical_storage_buffer64_vulkan
func.func @module_physical_storage_buffer64_vulkan() {
  // CHECK: spv.module min version: v1.0
  // CHECK: spv.module max version: v1.5
  // CHECK: spv.module extensions: [ [SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer] [SPV_KHR_vulkan_memory_model] ]
  // CHECK: spv.module capabilities: [ [PhysicalStorageBufferAddresses] [VulkanMemoryModel] ]
  spv.module PhysicalStorageBuffer64 Vulkan { }
  return
}
