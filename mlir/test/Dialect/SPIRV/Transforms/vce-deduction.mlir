// RUN: mlir-opt -spirv-update-vce %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

// Test deducing minimal version.
// spv.IAdd is available from v1.0.

// CHECK: vce_triple = #spv.vce<v1.0, [Shader], []>
spv.module "Logical" "GLSL450" {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}

// Test deducing minimal version.
// spv.GroupNonUniformBallot is available since v1.3.

// CHECK: vce_triple = #spv.vce<v1.3, [GroupNonUniformBallot, Shader], []>
spv.module "Logical" "GLSL450" {
  spv.func @group_non_uniform_ballot(%predicate : i1) -> vector<4xi32> "None" {
    %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader, GroupNonUniformBallot], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}

//===----------------------------------------------------------------------===//
// Capability
//===----------------------------------------------------------------------===//

// Test minimal capabilities.

// CHECK: vce_triple = #spv.vce<v1.0, [Shader], []>
spv.module "Logical" "GLSL450" {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, Float16, Float64, Int16, Int64, VariablePointers], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}

// Test deducing implied capability.
// AtomicStorage implies Shader.

// CHECK: vce_triple = #spv.vce<v1.0, [Shader], []>
spv.module "Logical" "GLSL450" {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [AtomicStorage], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
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

// CHECK: vce_triple = #spv.vce<v1.3, [GroupNonUniformArithmetic, Shader], []>
spv.module "Logical" "GLSL450" {
  spv.func @group_non_uniform_iadd(%val : i32) -> i32 "None" {
    %0 = spv.GroupNonUniformIAdd "Subgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, GroupNonUniformArithmetic], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}

// CHECK: vce_triple = #spv.vce<v1.3, [GroupNonUniformClustered, GroupNonUniformBallot, Shader], []>
spv.module "Logical" "GLSL450" {
  spv.func @group_non_uniform_iadd(%val : i32) -> i32 "None" {
    %0 = spv.GroupNonUniformIAdd "Subgroup" "Reduce" %val : i32
    spv.ReturnValue %0: i32
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.3, [Shader, GroupNonUniformClustered, GroupNonUniformBallot], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//

// Test deducing minimal extensions.
// spv.SubgroupBallotKHR requires the SPV_KHR_shader_ballot extension.

// CHECK: vce_triple = #spv.vce<v1.0, [SubgroupBallotKHR, Shader], [SPV_KHR_shader_ballot]>
spv.module "Logical" "GLSL450" {
  spv.func @subgroup_ballot(%predicate : i1) -> vector<4xi32> "None" {
    %0 = spv.SubgroupBallotKHR %predicate: vector<4xi32>
    spv.ReturnValue %0: vector<4xi32>
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, SubgroupBallotKHR],
             [SPV_KHR_shader_ballot, SPV_KHR_shader_clock, SPV_KHR_variable_pointers]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}

// Test deducing implied extension.
// Vulkan memory model requires SPV_KHR_vulkan_memory_model, which is enabled
// implicitly by v1.5.

// CHECK: vce_triple = #spv.vce<v1.0, [VulkanMemoryModel], [SPV_KHR_vulkan_memory_model]>
spv.module "Logical" "Vulkan" {
  spv.func @iadd(%val : i32) -> i32 "None" {
    %0 = spv.IAdd %val, %val: i32
    spv.ReturnValue %0: i32
  }
} attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.5, [Shader, VulkanMemoryModel], []>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
}
