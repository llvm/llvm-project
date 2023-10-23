// RUN: mlir-opt %s --spirv-attach-target='module=spirv.* ver=v1.0 caps=Kernel' | FileCheck %s

module attributes {gpu.container_module} {
// CHECK: @spirv_module_1 [#spirv.target<vce = #spirv.vce<v1.0, [Kernel], []>, resource_limits = <>>]
gpu.module @spirv_module_1 {
}
}
