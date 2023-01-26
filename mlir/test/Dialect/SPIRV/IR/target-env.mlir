// RUN: mlir-opt -mlir-disable-threading -test-spirv-target-env %s | FileCheck %s

// Note: The following tests check that a spirv.target_env can properly control
// the conversion target and filter unavailable ops during the conversion.
// We don't care about the op argument consistency too much; so certain enum
// values for enum attributes may not make much sense for the test op.

// spirv.AtomicCompareExchangeWeak is available from SPIR-V 1.0 to 1.3 under
// Kernel capability.
// spirv.AtomicCompareExchangeWeak has two memory semantics enum attribute,
// whose value, if containing AtomicCounterMemory bit, additionally requires
// AtomicStorage capability.

// spirv.BitReverse is available in all SPIR-V versions under Shader capability.

// spirv.GroupNonUniformBallot is available starting from SPIR-V 1.3 under
// GroupNonUniform capability.

// spirv.KHR.SubgroupBallot is available under in all SPIR-V versions under
// SubgroupBallotKHR capability and SPV_KHR_shader_ballot extension.

// Integer Dot Product ops (spirv.*Dot*) require the
// SPV_KHR_integer_dot_product extension and a number of related capabilities.

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
func.func @cmp_exchange_weak_suitable_version_capabilities(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.1, [Kernel, AtomicStorage], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.AtomicCompareExchangeWeak "Workgroup" "AcquireRelease|AtomicCounterMemory" "Acquire"
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spirv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @cmp_exchange_weak_unsupported_version
func.func @cmp_exchange_weak_unsupported_version(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, AtomicStorage], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spirv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

//===----------------------------------------------------------------------===//
// MinVersion
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @group_non_uniform_ballot_suitable_version
func.func @group_non_uniform_ballot_suitable_version(%predicate: i1) -> vector<4xi32> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.GroupNonUniformBallot <Workgroup>
  %0 = "test.convert_to_group_non_uniform_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @group_non_uniform_ballot_unsupported_version
func.func @group_non_uniform_ballot_unsupported_version(%predicate: i1) -> vector<4xi32> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.1, [GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_group_non_uniform_ballot_op
  %0 = "test.convert_to_group_non_uniform_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

//===----------------------------------------------------------------------===//
// Capability
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmp_exchange_weak_missing_capability_kernel
func.func @cmp_exchange_weak_missing_capability_kernel(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [AtomicStorage], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spirv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @cmp_exchange_weak_missing_capability_atomic_storage
func.func @cmp_exchange_weak_missing_capability_atomic_storage(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_atomic_compare_exchange_weak_op
  %0 = "test.convert_to_atomic_compare_exchange_weak_op"(%ptr, %value, %comparator): (!spirv.ptr<i32, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @subgroup_ballot_missing_capability
func.func @subgroup_ballot_missing_capability(%predicate: i1) -> vector<4xi32> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [], [SPV_KHR_shader_ballot]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_subgroup_ballot_op
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @bit_reverse_directly_implied_capability
func.func @bit_reverse_directly_implied_capability(%operand: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Geometry], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.BitReverse
  %0 = "test.convert_to_bit_reverse_op"(%operand): (i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @bit_reverse_recursively_implied_capability
func.func @bit_reverse_recursively_implied_capability(%operand: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [GeometryPointSize], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.BitReverse
  %0 = "test.convert_to_bit_reverse_op"(%operand): (i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sdot_scalar_i32_i32_capabilities
func.func @sdot_scalar_i32_i32_capabilities(%operand: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInput4x8BitPacked], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.SDot
  %0 = "test.convert_to_sdot_op"(%operand, %operand) {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sdot_scalar_i32_i32_missing_capability1
func.func @sdot_scalar_i32_i32_missing_capability1(%operand: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_sdot_op
  %0 = "test.convert_to_sdot_op"(%operand, %operand) {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sdot_scalar_i32_i32_missing_capability2
func.func @sdot_scalar_i32_i32_missing_capability2(%operand: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProductInput4x8BitPacked], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_sdot_op
  %0 = "test.convert_to_sdot_op"(%operand, %operand) {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sudot_vector_4xi8_i32_capabilities
func.func @sudot_vector_4xi8_i32_capabilities(%operand: vector<4xi8>) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInput4x8Bit], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.SUDot
  %0 = "test.convert_to_sudot_op"(%operand, %operand): (vector<4xi8>, vector<4xi8>) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sudot_vector_4xi8_i32_missing_capability1
func.func @sudot_vector_4xi8_i32_missing_capability1(%operand: vector<4xi8>) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_sudot_op
  %0 = "test.convert_to_sudot_op"(%operand, %operand): (vector<4xi8>, vector<4xi8>) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sudot_vector_4xi8_i32_missing_capability2
func.func @sudot_vector_4xi8_i32_missing_capability2(%operand: vector<4xi8>) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProductInput4x8Bit], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_sudot_op
  %0 = "test.convert_to_sudot_op"(%operand, %operand): (vector<4xi8>, vector<4xi8>) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @udot_vector_4xi16_i64_capabilities
func.func @udot_vector_4xi16_i64_capabilities(%operand: vector<4xi16>) -> i64 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInputAll, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.UDot
  %0 = "test.convert_to_udot_op"(%operand, %operand): (vector<4xi16>, vector<4xi16>) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @udot_vector_4xi16_i64_missing_capability1
func.func @udot_vector_4xi16_i64_missing_capability1(%operand: vector<4xi16>) -> i64 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_op
  %0 = "test.convert_to_udot_op"(%operand, %operand): (vector<4xi16>, vector<4xi16>) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @udot_vector_4xi16_i64_missing_capability2
func.func @udot_vector_4xi16_i64_missing_capability2(%operand: vector<4xi16>) -> i64 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProductInputAll, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_op
  %0 = "test.convert_to_udot_op"(%operand, %operand): (vector<4xi16>, vector<4xi16>) -> (i64)
  return %0 : i64
}

// CHECK-LABEL: @sdot_acc_sat_scalar_i32_i32_capabilities
func.func @sdot_acc_sat_scalar_i32_i32_capabilities(%operand: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInput4x8BitPacked], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.SDotAccSat
  %0 = "test.convert_to_sdot_acc_sat_op"(%operand, %operand, %operand)
         {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @sudot_acc_sat_vector_4xi8_i32_capabilities
func.func @sudot_acc_sat_vector_4xi8_i32_capabilities(%operand: vector<4xi8>, %acc: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInput4x8Bit], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.SUDotAccSat
  %0 = "test.convert_to_sudot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi8>, vector<4xi8>, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @udot_acc_sat_vector_4xi8_i32_missing_capability1
func.func @udot_acc_sat_vector_4xi8_i32_missing_capability1(%operand: vector<4xi8>, %acc: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInputAll, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_acc_sat_op
  %0 = "test.convert_to_udot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi8>, vector<4xi8>, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @udot_acc_sat_vector_4xi8_i32_missing_capability2
func.func @udot_acc_sat_vector_4xi8_i32_missing_capability2(%operand: vector<4xi8>, %acc: i32) -> i32 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProductInputAll, DotProductInput4x8Bit, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_acc_sat_op
  %0 = "test.convert_to_udot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi8>, vector<4xi8>, i32) -> (i32)
  return %0: i32
}

// CHECK-LABEL: @udot_acc_sat_vector_4xi16_i64_capabilities
func.func @udot_acc_sat_vector_4xi16_i64_capabilities(%operand: vector<4xi16>, %acc: i64) -> i64 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, DotProductInputAll, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.UDotAccSat
  %0 = "test.convert_to_udot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi16>, vector<4xi16>, i64) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @udot_acc_sat_vector_4xi16_i64_missing_capability1
func.func @udot_acc_sat_vector_4xi16_i64_missing_capability1(%operand: vector<4xi16>, %acc: i64) -> i64 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProductInputAll, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_acc_sat_op
  %0 = "test.convert_to_udot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi16>, vector<4xi16>, i64) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @udot_acc_sat_vector_4xi16_i64_missing_capability2
func.func @udot_acc_sat_vector_4xi16_i64_missing_capability2(%operand: vector<4xi16>, %acc: i64) -> i64 attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
    [DotProduct, Int16, Int64], [SPV_KHR_integer_dot_product]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_acc_sat_op
  %0 = "test.convert_to_udot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi16>, vector<4xi16>, i64) -> (i64)
  return %0: i64
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @subgroup_ballot_suitable_extension
func.func @subgroup_ballot_suitable_extension(%predicate: i1) -> vector<4xi32> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [SubgroupBallotKHR], [SPV_KHR_shader_ballot]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.KHR.SubgroupBallot
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @subgroup_ballot_missing_extension
func.func @subgroup_ballot_missing_extension(%predicate: i1) -> vector<4xi32> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [SubgroupBallotKHR], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_subgroup_ballot_op
  %0 = "test.convert_to_subgroup_ballot_op"(%predicate): (i1) -> (vector<4xi32>)
  return %0: vector<4xi32>
}

// CHECK-LABEL: @module_suitable_extension1
func.func @module_suitable_extension1() attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_vulkan_memory_model, SPV_EXT_physical_storage_buffer]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.module PhysicalStorageBuffer64 Vulkan
  "test.convert_to_module_op"() : () ->()
  return
}

// CHECK-LABEL: @module_suitable_extension2
func.func @module_suitable_extension2() attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_vulkan_memory_model, SPV_KHR_physical_storage_buffer]>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.module PhysicalStorageBuffer64 Vulkan
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @module_missing_extension_mm
func.func @module_missing_extension_mm() attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_physical_storage_buffer]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_module_op
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @module_missing_extension_am
func.func @module_missing_extension_am() attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [VulkanMemoryModel, PhysicalStorageBufferAddresses], [SPV_KHR_vulkan_memory_model]>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_module_op
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @module_implied_extension
func.func @module_implied_extension() attributes {
  // Version 1.5 implies SPV_KHR_vulkan_memory_model and SPV_KHR_physical_storage_buffer.
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [VulkanMemoryModel, PhysicalStorageBufferAddresses], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.module PhysicalStorageBuffer64 Vulkan
  "test.convert_to_module_op"() : () -> ()
  return
}

// CHECK-LABEL: @udot_vector_4xi16_i64_implied_extension
func.func @udot_vector_4xi16_i64_implied_extension(%operand: vector<4xi16>) -> i64 attributes {
  // Version 1.6 implies SPV_KHR_integer_to_product.
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
    [DotProduct, DotProductInputAll, Int16, Int64], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.UDot
  %0 = "test.convert_to_udot_op"(%operand, %operand): (vector<4xi16>, vector<4xi16>) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @udot_vector_4xi16_i64_missing_extension
func.func @udot_vector_4xi16_i64_missing_extension(%operand: vector<4xi16>) -> i64 attributes {
  // Version 1.5 does not imply SPV_KHR_integer_to_product.
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5,
    [DotProduct, DotProductInputAll, Int16, Int64], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_udot_op
  %0 = "test.convert_to_udot_op"(%operand, %operand): (vector<4xi16>, vector<4xi16>) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @sdot_acc_sat_vector_4xi16_i64_implied_extension
func.func @sdot_acc_sat_vector_4xi16_i64_implied_extension(%operand: vector<4xi16>, %acc: i64) -> i64 attributes {
  // Version 1.6 implies SPV_KHR_integer_to_product.
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.6,
    [DotProduct, DotProductInputAll, Int16, Int64], []>, #spirv.resource_limits<>>
} {
  // CHECK: spirv.SDotAccSat
  %0 = "test.convert_to_sdot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi16>, vector<4xi16>, i64) -> (i64)
  return %0: i64
}

// CHECK-LABEL: @sdot_acc_sat_vector_4xi16_i64_missing_extension
func.func @sdot_acc_sat_vector_4xi16_i64_missing_extension(%operand: vector<4xi16>, %acc: i64) -> i64 attributes {
  // Version 1.5 does not imply SPV_KHR_integer_to_product.
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5,
    [DotProduct, DotProductInputAll, Int16, Int64], []>, #spirv.resource_limits<>>
} {
  // CHECK: test.convert_to_sdot_acc_sat_op
  %0 = "test.convert_to_sdot_acc_sat_op"(%operand, %operand, %acc): (vector<4xi16>, vector<4xi16>, i64) -> (i64)
  return %0: i64
}
