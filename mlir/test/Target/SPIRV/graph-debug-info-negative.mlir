// RUN: mlir-translate %s --test-spirv-roundtrip-debug --no-implicit-module --verify-diagnostics --split-input-file

// expected-error@below {{SPV_KHR_non_semantic_info extension not available}}
spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
}

// -----

// expected-error @+1 {{SPV_KHR_non_semantic_info extension not available}}
spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  spirv.ARM.Graph @g(%arg0: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1xi8>) {
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<1xi8>
  }
}
