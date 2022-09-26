// RUN: mlir-opt -mlir-disable-threading -test-spirv-op-availability %s | FileCheck %s

// CHECK-LABEL: iadd
func.func @iadd(%arg: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.6
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ ]
  %0 = spirv.IAdd %arg, %arg: i32
  return %0: i32
}

// CHECK: atomic_compare_exchange_weak
func.func @atomic_compare_exchange_weak(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: min version: v1.0
  // CHECK: max version: v1.3
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [Kernel] ]
  %0 = spirv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spirv.ptr<i32, Workgroup>
  return %0: i32
}

// CHECK-LABEL: subgroup_ballot
func.func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: min version: v1.3
  // CHECK: max version: v1.6
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [GroupNonUniformBallot] ]
  %0 = spirv.GroupNonUniformBallot <Workgroup> %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// CHECK-LABEL: module_logical_glsl450
func.func @module_logical_glsl450() {
  // CHECK: spirv.module min version: v1.0
  // CHECK: spirv.module max version: v1.6
  // CHECK: spirv.module extensions: [ ]
  // CHECK: spirv.module capabilities: [ [Shader] ]
  spirv.module Logical GLSL450 { }
  return
}

// CHECK-LABEL: module_physical_storage_buffer64_vulkan
func.func @module_physical_storage_buffer64_vulkan() {
  // CHECK: spirv.module min version: v1.0
  // CHECK: spirv.module max version: v1.6
  // CHECK: spirv.module extensions: [ [SPIRV_EXT_physical_storage_buffer, SPIRV_KHR_physical_storage_buffer] [SPIRV_KHR_vulkan_memory_model] ]
  // CHECK: spirv.module capabilities: [ [PhysicalStorageBufferAddresses] [VulkanMemoryModel] ]
  spirv.module PhysicalStorageBuffer64 Vulkan { }
  return
}
