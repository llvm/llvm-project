// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: @module_1 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>>]
gpu.module @module_1 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>>] {
}

// CHECK: @module_2 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>, client_api = OpenCL>]
gpu.module @module_2 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>, client_api = OpenCL>] {
}

// CHECK: @module_3 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>, client_api = OpenCL, vendor_id = Intel, device_type = IntegratedGPU>]
gpu.module @module_3 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>, client_api = OpenCL, vendor_id = Intel, device_type = IntegratedGPU>] {
}

