// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: @module_1 [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>]
gpu.module @module_1 [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>] {
}

// CHECK: @module_2 [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, api=OpenCL, #spirv.resource_limits<>>]
gpu.module @module_2 [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, api=OpenCL, #spirv.resource_limits<>>] {
}

// CHECK: @module_3 [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, api=OpenCL, Intel:IntegratedGPU, #spirv.resource_limits<>>]
gpu.module @module_3 [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, api=OpenCL, Intel:IntegratedGPU, #spirv.resource_limits<>>] {
}

