// RUN: mlir-translate %s --test-spirv-roundtrip-debug --no-implicit-module --verify-diagnostics

// expected-error@below {{SPV_KHR_non_semantic_info extension not available}}
spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {
}
