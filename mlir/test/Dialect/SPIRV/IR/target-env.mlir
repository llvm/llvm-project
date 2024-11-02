// RUN: mlir-opt -mlir-disable-threading -test-spirv-target-env %s | FileCheck %s

// Note: The following tests check that a spv.target_env can properly control
// the conversion target and filter unavailable ops during the conversion.
// We don't care about the op argument consistency too much; so certain enum
// values for enum attributes may not make much sense for the test op.

// spv.AtomicCompareExchangeWeak is available from SPIR-V 1.0 to 1.3 under
// Kernel capability.
// spv.AtomicCompareExchangeWeak has two memory semantics enum attribute,
// whose value, if containing AtomicCounterMemory bit, additionally requires
// AtomicStorage capability.

// spv.BitReverse is available in all SPIR-V versions under Shader capability.

// spv.GroupNonUniformBallot is available starting from SPIR-V 1.3 under
// GroupNonUniform capability.

// spv.SubgroupBallotKHR is available under in all SPIR-V versions under
// SubgroupBallotKHR capability and SPV_KHR_shader_ballot extension.

// The GeometryPointSize capability implies the Geometry capability, which
// implies the Shader capability.

// PhysicalStorageBuffer64 addressing model is available via extension
// SPV_EXT_physical_storage_buffer or SPV_KHR_physical_storage_buffer;
// both extensions are incorporated into SPIR-V 1.5.

// Vulkan memory model is available via extension SPV_KHR_vulkan_memory_model,
// which extensions are incorporated into SPIR-V 1.5.

//===----------------------------------------------------------------------===//
// MaxVersion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmp_exchange_weak_suitable_version_capabilities
func.func @cmp_exchange_weak_suitable_version_capabilities(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.1, [Kernel, AtomicStorage], []>, {}>
} {
  // CHECK: spv.AtomicCompareExchangeWeak "Workgroup" "AcquireRelease|AtomicCounterMemory" "Acquire"
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @cmp_exchange_weak_unsupported_version
func.func @cmp_exchange_weak_unsupported_version(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.4, [Kernel, AtomicStorage], []>, {}>
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

//===----------------------------------------------------------------------===//
// MinVersion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_ballot_suitable_version
func.func @group_non_uniform_ballot_suitable_version(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.4, [GroupNonUniformBallot], []>, {}>
} {
  // CHECK: spv.GroupNonUniformBallot Workgroup
  %0 = "test.convert_to_group_non_uniform_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @group_non_uniform_ballot_unsupported_version
func.func @group_non_uniform_ballot_unsupported_version(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.1, [GroupNonUniformBallot], []>, {}>
} {
  // CHECK: test.convert_to_group_non_uniform_ballot_op
  %0 = "test.convert_to_group_non_uniform_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

//===----------------------------------------------------------------------===//
// Capability
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmp_exchange_weak_missing_capability_kernel
func.func @cmp_exchange_weak_missing_capability_kernel(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.3, [AtomicStorage], []>, {}>
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @cmp_exchange_weak_missing_capability_atomic_storage
func.func @cmp_exchange_weak_missing_capability_atomic_storage(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.3, [Kernel], []>, {}>
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @subgroup_ballot_missing_capability
func.func @subgroup_ballot_missing_capability(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.4, [], [SPV_KHR_shader_ballot]>, {}>
} {
  // CHECK: test.convert_to_subgroup_ballot_op
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @bit_reverse_directly_implied_capability
func.func @bit_reverse_directly_implied_capability(%operand: i32) -> i32 attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Geometry], []>, {}>
} {
  // CHECK: spv.BitReverse
  %0 = "test.convert_to_bit_reverse_op"(%operand): (i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @bit_reverse_recursively_implied_capability
func.func @bit_reverse_recursively_implied_capability(%operand: i32) -> i32 attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [GeometryPointSize], []>, {}>
} {
  // CHECK: spv.BitReverse
  %0 = "test.convert_to_bit_reverse_op"(%operand): (i32) -> (i32)
  return %0: i32
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @subgroup_ballot_suitable_extension
func.func @subgroup_ballot_suitable_extension(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.4, [SubgroupBallotKHR], [SPV_KHR_shader_ballot]>, {}>
} {
  // CHECK: spv.SubgroupBallotKHR
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @subgroup_ballot_missing_extension
func.func @subgroup_ballot_missing_extension(%predicate: i1) -> vector<4xi32> attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.4, [SubgroupBallotKHR], []>, {}>
} {
  // CHECK: test.convert_to_subgroup_ballot_op
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @module_suitable_extension1
func.func @module_suitable_extension1() attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_vulkan_memory_model, SPV_EXT_physical_storage_buffer]>, {}>
} {
  // CHECK: spv.module PhysicalStorageBuffer64 Vulkan
  "test.convert_to_module_op"() : () ->()
  return
}

// CHECK-LABEL: @module_suitable_extension2
func.func @module_suitable_extension2() attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_vulkan_memory_model, SPV_KHR_physical_storage_buffer]>, {}>
} {
  // CHECK: spv.module PhysicalStorageBuffer64 Vulkan
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @module_missing_extension_mm
func.func @module_missing_extension_mm() attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]>, {}>
} {
  // CHECK: test.convert_to_module_op
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @module_missing_extension_am
func.func @module_missing_extension_am() attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_vulkan_memory_model]>, {}>
} {
  // CHECK: test.convert_to_module_op
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @module_implied_extension
func.func @module_implied_extension() attributes {
  // Version 1.5 implies SPV_KHR_vulkan_memory_model and SPV_KHR_physical_storage_buffer.
  spv.target_env = #spv.target_env<#spv.vce<v1.5, [VulkanMemoryModel, PhysicalStorageBufferAddresses], []>, {}>
} {
  // CHECK: spv.module PhysicalStorageBuffer64 Vulkan
  "test.convert_to_module_op"() : () -> ()
  return
}
