// RUN: mlir-opt %s --gpu-module-to-binary | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.binary @kernel_module
  // CHECK:[#gpu.object<#spirv.target_env<#spirv.vce<v1.0, [Int64, Int16, Kernel, Addresses], []>, #spirv.resource_limits<>>, "{{.*}}">]
  gpu.module @kernel_module [#spirv.target_env<#spirv.vce<v1.0, [Int64, Int16, Kernel, Addresses], []>, #spirv.resource_limits<>>] {
    spirv.module @__spv__kernel_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Int16, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_AMD_shader_ballot]>, api=OpenCL, #spirv.resource_limits<>>} {
      spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>) "None" attributes {workgroup_attributions = 0 : i64} {
        spirv.Return
      }
    }
  }
}
