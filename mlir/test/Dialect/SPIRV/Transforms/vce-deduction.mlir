// RUN: mlir-opt -spirv-update-vce %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

// Test deducing minimal version.
// spv.IAdd is available from v1.0.

// CHECK: requires #spv.vce<v1.0, [Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader], []>, {}>
} {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
}

// Test deducing minimal version.
// spv.GroupNonUniformBallot is available since v1.3.

// CHECK: requires #spv.vce<v1.3, [GroupNonUniformBallot, Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader, GroupNonUniformBallot], []>, {}>
} {
  spv.func @group_non_uniform_ballot(%predicate : i1) -> vector<4xi32> "None" {
    %0 = spv.GroupNonUniformBallot Workgroup %predicate : vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
}

//===----------------------------------------------------------------------===//
// Capability
//===----------------------------------------------------------------------===//

// Test minimal capabilities.

// CHECK: requires #spv.vce<v1.0, [Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, Float16, Float64, Int16, Int64, VariablePointers], []>, {}>
} {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
}

// Test deducing implied capability.
// AtomicStorage implies Shader.

// CHECK: requires #spv.vce<v1.0, [Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [AtomicStorage], []>, {}>
} {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
}

// Test selecting the capability available in the target environment.
// spv.GroupNonUniform op itself can be enabled via any of
// * GroupNonUniformArithmetic
// * GroupNonUniformClustered
// * GroupNonUniformPartitionedNV
// Its 'Reduce' group operation can be enabled via any of
// * Kernel
// * GroupNonUniformArithmetic
// * GroupNonUniformBallot

// CHECK: requires #spv.vce<v1.3, [GroupNonUniformArithmetic, Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>, {}>
} {
  spv.func @group_non_uniform_iadd(%val : i32) -> i32 "None" {
    %0 = spv.GroupNonUniformIAdd "Subgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }
}

// CHECK: requires #spv.vce<v1.3, [GroupNonUniformClustered, GroupNonUniformBallot, Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, GroupNonUniformClustered, GroupNonUniformBallot], []>, {}>
} {
  spv.func @group_non_uniform_iadd(%val : i32) -> i32 "None" {
    %0 = spv.GroupNonUniformIAdd "Subgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }
}

// Test type required capabilities

// Using 8-bit integers in non-interface storage class requires Int8.
// CHECK: requires #spv.vce<v1.0, [Int8, Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, Int8], []>, {}>
} {
  spv.func @iadd_function(%val : i8) -> i8 "None" {
    %0 = spv.IAdd %val, %val : i8
    spv.ReturnValue %0: i8
  }
}

// Using 16-bit floats in non-interface storage class requires Float16.
// CHECK: requires #spv.vce<v1.0, [Float16, Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, Float16], []>, {}>
} {
  spv.func @fadd_function(%val : f16) -> f16 "None" {
    %0 = spv.FAdd %val, %val : f16
    spv.ReturnValue %0: f16
  }
}

// Using 16-element vectors requires Vector16.
// CHECK: requires #spv.vce<v1.0, [Vector16, Shader], []>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, Vector16], []>, {}>
} {
  spv.func @iadd_v16_function(%val : vector<16xi32>) -> vector<16xi32> "None" {
    %0 = spv.IAdd %val, %val : vector<16xi32>
    spv.ReturnValue %0: vector<16xi32>
  }
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//

// Test deducing minimal extensions.
// spv.SubgroupBallotKHR requires the SPV_KHR_shader_ballot extension.

// CHECK: requires #spv.vce<v1.0, [SubgroupBallotKHR, Shader], [SPV_KHR_shader_ballot]>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, SubgroupBallotKHR],
             [SPV_KHR_shader_ballot, SPV_KHR_shader_clock, SPV_KHR_variable_pointers]>, {}>
} {
  spv.func @subgroup_ballot(%predicate : i1) -> vector<4xi32> "None" {
    %0 = spv.SubgroupBallotKHR %predicate: vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
}

// Test deducing implied extension.
// Vulkan memory model requires SPV_KHR_vulkan_memory_model, which is enabled
// implicitly by v1.5.

// CHECK: requires #spv.vce<v1.0, [VulkanMemoryModel], [SPV_KHR_vulkan_memory_model]>
spv.module Logical Vulkan attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader, VulkanMemoryModel], []>, {}>
} {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
}

// Test type required extensions

// Using 8-bit integers in interface storage class requires additional
// extensions and capabilities.
// CHECK: requires #spv.vce<v1.0, [StorageBuffer16BitAccess, Shader, Int16], [SPV_KHR_16bit_storage, SPV_KHR_storage_buffer_storage_class]>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, StorageBuffer16BitAccess, Int16], []>, {}>
} {
  spv.func @iadd_storage_buffer(%ptr : !spv.ptr<i16, StorageBuffer>) -> i16 "None" {
    %0 = spv.Load "StorageBuffer" %ptr : i16
    %1 = spv.IAdd %0, %0 : i16
    spv.ReturnValue %1: i16
  }
}

// Complicated nested types
// * Buffer requires ImageBuffer or SampledBuffer.
// * Rg32f requires StorageImageExtendedFormats.
// CHECK: requires #spv.vce<v1.0, [UniformAndStorageBuffer8BitAccess, StorageUniform16, Int64, Shader, ImageBuffer, StorageImageExtendedFormats], [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>
spv.module Logical GLSL450 attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader, UniformAndStorageBuffer8BitAccess, StorageBuffer16BitAccess, StorageUniform16, Int16, Int64, ImageBuffer, StorageImageExtendedFormats], []>,
    {}>
} {
  spv.GlobalVariable @data : !spv.ptr<!spv.struct<(i8 [0], f16 [2], i64 [4])>, Uniform>
  spv.GlobalVariable @img  : !spv.ptr<!spv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Rg32f>, UniformConstant>
}
