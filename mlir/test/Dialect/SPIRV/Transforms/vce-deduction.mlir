// RUN: mlir-opt -spirv-update-vce %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

// Test deducing minimal version.
// spirv.IAdd is available from v1.0.

// CHECK: requires #spirv.vce<v1.0, [Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.5, [Shader], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spirv.IAdd %val, %val: i32
    spirv.ReturnValue %0: i32
  }
}

// Test deducing minimal version.
// spirv.GroupNonUniformBallot is available since v1.3.

// CHECK: requires #spirv.vce<v1.3, [GroupNonUniformBallot, Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.5, [Shader, GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {
  spirv.func @group_non_uniform_ballot(%predicate : i1) -> vector<4xi32> "None" {
    %0 = spirv.GroupNonUniformBallot <Workgroup> %predicate : vector<4xi32>
    spirv.ReturnValue %0: vector<4xi32>
  }
}

//===----------------------------------------------------------------------===//
// Capability
//===----------------------------------------------------------------------===//

// Test minimal capabilities.

// CHECK: requires #spirv.vce<v1.0, [Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Float16, Float64, Int16, Int64, VariablePointers], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spirv.IAdd %val, %val: i32
    spirv.ReturnValue %0: i32
  }
}

// Test Physical Storage Buffers are deduced correctly.

// CHECK: spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.0, [PhysicalStorageBufferAddresses, Shader], [SPV_EXT_physical_storage_buffer]>
spirv.module PhysicalStorageBuffer64 GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, PhysicalStorageBufferAddresses], [SPV_EXT_physical_storage_buffer]>, #spirv.resource_limits<>>
} {
  spirv.func @physical_ptr(%val : !spirv.ptr<f32, PhysicalStorageBuffer>) "None" {
    spirv.Return
  }
}

// Test deducing implied capability.
// AtomicStorage implies Shader.

// CHECK: requires #spirv.vce<v1.0, [Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [AtomicStorage], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spirv.IAdd %val, %val: i32
    spirv.ReturnValue %0: i32
  }
}

// Test selecting the capability available in the target environment.
// spirv.GroupNonUniform op itself can be enabled via any of
// * GroupNonUniformArithmetic
// * GroupNonUniformClustered
// * GroupNonUniformPartitionedNV
// Its 'Reduce' group operation can be enabled via any of
// * Kernel
// * GroupNonUniformArithmetic
// * GroupNonUniformBallot

// CHECK: requires #spirv.vce<v1.3, [GroupNonUniformArithmetic, Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>, #spirv.resource_limits<>>
} {
  spirv.func @group_non_uniform_iadd(%val : i32) -> i32 "None" {
    %0 = spirv.GroupNonUniformIAdd "Subgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }
}

// CHECK: requires #spirv.vce<v1.3, [GroupNonUniformClustered, GroupNonUniformBallot, Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniformClustered, GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {
  spirv.func @group_non_uniform_iadd(%val : i32) -> i32 "None" {
    %0 = spirv.GroupNonUniformIAdd "Subgroup" "Reduce" %val : i32
    spirv.ReturnValue %0: i32
  }
}

// Test type required capabilities

// Using 8-bit integers in non-interface storage class requires Int8.
// CHECK: requires #spirv.vce<v1.0, [Int8, Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, Int8], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd_function(%val : i8) -> i8 "None" {
    %0 = spirv.IAdd %val, %val : i8
    spirv.ReturnValue %0: i8
  }
}

// Using 16-bit floats in non-interface storage class requires Float16.
// CHECK: requires #spirv.vce<v1.0, [Float16, Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, Float16], []>, #spirv.resource_limits<>>
} {
  spirv.func @fadd_function(%val : f16) -> f16 "None" {
    %0 = spirv.FAdd %val, %val : f16
    spirv.ReturnValue %0: f16
  }
}

// Using 16-element vectors requires Vector16.
// CHECK: requires #spirv.vce<v1.0, [Vector16, Shader], []>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, Vector16], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd_v16_function(%val : vector<16xi32>) -> vector<16xi32> "None" {
    %0 = spirv.IAdd %val, %val : vector<16xi32>
    spirv.ReturnValue %0: vector<16xi32>
  }
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//

// Test deducing minimal extensions.
// spirv.KHR.SubgroupBallot requires the SPV_KHR_shader_ballot extension.

// CHECK: requires #spirv.vce<v1.0, [SubgroupBallotKHR, Shader], [SPV_KHR_shader_ballot]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, SubgroupBallotKHR],
             [SPV_KHR_shader_ballot, SPV_KHR_shader_clock, SPV_KHR_variable_pointers]>, #spirv.resource_limits<>>
} {
  spirv.func @subgroup_ballot(%predicate : i1) -> vector<4xi32> "None" {
    %0 = spirv.KHR.SubgroupBallot %predicate: vector<4xi32>
    spirv.ReturnValue %0: vector<4xi32>
  }
}

// Test deducing implied extension.
// Vulkan memory model requires SPV_KHR_vulkan_memory_model, which is enabled
// implicitly by v1.5.

// CHECK: requires #spirv.vce<v1.0, [VulkanMemoryModel], [SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.5, [Shader, VulkanMemoryModel], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spirv.IAdd %val, %val: i32
    spirv.ReturnValue %0: i32
  }
}

// Test type required extensions

// Using 8-bit integers in interface storage class requires additional
// extensions and capabilities.
// CHECK: requires #spirv.vce<v1.0, [StorageBuffer16BitAccess, Shader, Int16], [SPV_KHR_16bit_storage, SPV_KHR_storage_buffer_storage_class]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, StorageBuffer16BitAccess, Int16], []>, #spirv.resource_limits<>>
} {
  spirv.func @iadd_storage_buffer(%ptr : !spirv.ptr<i16, StorageBuffer>) -> i16 "None" {
    %0 = spirv.Load "StorageBuffer" %ptr : i16
    %1 = spirv.IAdd %0, %0 : i16
    spirv.ReturnValue %1: i16
  }
}

// Complicated nested types
// * Buffer requires ImageBuffer or SampledBuffer.
// * Rg32f requires StorageImageExtendedFormats.
// CHECK: requires #spirv.vce<v1.0, [UniformAndStorageBuffer8BitAccess, StorageUniform16, Int64, Shader, ImageBuffer, StorageImageExtendedFormats], [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.5, [Shader, UniformAndStorageBuffer8BitAccess, StorageBuffer16BitAccess, StorageUniform16, Int16, Int64, ImageBuffer, StorageImageExtendedFormats], []>,
    #spirv.resource_limits<>>
} {
  spirv.GlobalVariable @data : !spirv.ptr<!spirv.struct<(i8 [0], f16 [2], i64 [4])>, Uniform>
  spirv.GlobalVariable @img  : !spirv.ptr<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Rg32f>, UniformConstant>
}
