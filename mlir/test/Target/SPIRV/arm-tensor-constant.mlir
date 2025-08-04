// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s
// DISABLED: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv %s | spirv-val %}

// FIXME(#152012): Fix arm tensor constant validation errors and reenable spirv-val tests.

spirv.module Logical Vulkan requires #spirv.vce<v1.3,
             [VulkanMemoryModel, Shader, TensorsARM, Linkage], [SPV_KHR_vulkan_memory_model, SPV_ARM_tensors]> {
  // CHECK-LABEL: @arm_tensor_of_i32
  spirv.func @arm_tensor_of_i32() -> (!spirv.arm.tensor<2x3xi32>) "None" {
    // CHECK: {{%.*}} = spirv.Constant dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : !spirv.arm.tensor<2x3xi32>
    %0 = spirv.Constant dense<[[1, 2, 3], [4, 5, 6]]> : !spirv.arm.tensor<2x3xi32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xi32>
  }

  // CHECK-LABEL: @splat_arm_tensor_of_i32
  spirv.func @splat_arm_tensor_of_i32() -> (!spirv.arm.tensor<2x3xi32>) "None" {
    // CHECK: {{%.*}} = spirv.Constant dense<2> : !spirv.arm.tensor<2x3xi32>
    %0 = spirv.Constant dense<2> : !spirv.arm.tensor<2x3xi32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xi32>
  }

  // CHECK-LABEL: @arm_tensor_of_f32
  spirv.func @arm_tensor_of_f32() -> (!spirv.arm.tensor<2x3xf32>) "None" {
    // CHECK: {{%.*}} = spirv.Constant dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : !spirv.arm.tensor<2x3xf32>
    %0 = spirv.Constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>: !spirv.arm.tensor<2x3xf32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xf32>
  }

  // CHECK-LABEL: @splat_arm_tensor_of_f32
  spirv.func @splat_arm_tensor_of_f32() -> (!spirv.arm.tensor<2x3xf32>) "None" {
    // CHECK: {{%.*}} = spirv.Constant dense<2.000000e+00> : !spirv.arm.tensor<2x3xf32>
    %0 = spirv.Constant dense<2.0> : !spirv.arm.tensor<2x3xf32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xf32>
  }

  // CHECK-LABEL: @null_arm_tensor_of_i32
  spirv.func @null_arm_tensor_of_i32() -> (!spirv.arm.tensor<2x3xi32>) "None" {
    // CHECK: spirv.Constant dense<0> : !spirv.arm.tensor<2x3xi32>
    %0 = spirv.Constant dense<0> : !spirv.arm.tensor<2x3xi32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xi32>
  }

  // CHECK-LABEL: @null_arm_tensor_of_f32
  spirv.func @null_arm_tensor_of_f32() -> (!spirv.arm.tensor<2x3xf32>) "None" {
    // CHECK: spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<2x3xf32>
    %0 = spirv.Constant dense<0.0> : !spirv.arm.tensor<2x3xf32>
    spirv.ReturnValue %0 : !spirv.arm.tensor<2x3xf32>
  }
}
