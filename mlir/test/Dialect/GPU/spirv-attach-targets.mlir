// RUN: mlir-opt %s --spirv-attach-target='module=spirv.* ver=v1.0 caps=Kernel vendor=Intel device_type=Other' | FileCheck %s
// RUN: mlir-opt %s --spirv-attach-target='module=spirv_warm.* ver=v1.0 caps=Kernel vendor=Intel device_type=Other device_id=0' | FileCheck %s --check-prefix=CHECK_WARM

module attributes {gpu.container_module} {
//      CHECK: @spirv_hot_module [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, Intel:Other, #spirv.resource_limits<>>]
// CHECK_WARM: @spirv_hot_module {
gpu.module @spirv_hot_module {
}
//      CHECK: @spirv_warm_module [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, Intel:Other, #spirv.resource_limits<>>]
// CHECK_WARM: @spirv_warm_module [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, Intel:Other:0, #spirv.resource_limits<>>]
gpu.module @spirv_warm_module {
}
//      CHECK: @spirv_cold_module [#spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, Intel:Other, #spirv.resource_limits<>>]
// CHECK_WARM: @spirv_cold_module {
gpu.module @spirv_cold_module {
}
}
