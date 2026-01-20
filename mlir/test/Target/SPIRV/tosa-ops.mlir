// RUN: mlir-translate --no-implicit-module --split-input-file --test-spirv-roundtrip %s | FileCheck %s
// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module  --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @argmax_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<3x28x17x17xi8>, UniformConstant>
  spirv.GlobalVariable @argmax_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<3x28x17xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @argmax_int, @argmax_int_arg_0, @argmax_int_res_0
  spirv.ARM.Graph @argmax_int(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28x17xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
    %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x28x17xi32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28x17xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @argmax_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x2x7x14xf32>, UniformConstant>
  spirv.GlobalVariable @argmax_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x2x14xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @argmax_fp, @argmax_fp_arg_0, @argmax_fp_res_0
  spirv.ARM.Graph @argmax_fp(%arg0: !spirv.arm.tensor<2x2x7x14xf32>) -> (!spirv.arm.tensor<2x2x14xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
    %2 = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x2x14xi32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x2x14xi32>
  }
}
