// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv %s | spirv-val %}

// CHECK: spirv.module Logical Vulkan requires
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  // CHECK: spirv.GlobalVariable @main_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x16xf32>, UniformConstant>
  spirv.GlobalVariable @main_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x16xf32>, UniformConstant>
  // CHECK: spirv.GlobalVariable @main_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x16xf32>, UniformConstant>
  spirv.GlobalVariable @main_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x16xf32>, UniformConstant>
  // CHECK: spirv.GlobalVariable @main_res_0 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1x16xf32>, UniformConstant>
  spirv.GlobalVariable @main_res_0 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1x16xf32>, UniformConstant>
  // CHECK: spirv.ARM.GraphEntryPoint @main, @main_arg_0, @main_arg_1, @main_res_0
  spirv.ARM.GraphEntryPoint @main, @main_arg_0, @main_arg_1, @main_res_0
  // CHECK: spirv.ARM.Graph @main(%arg0: !spirv.arm.tensor<1x16xf32>, %arg1: !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
  spirv.ARM.Graph @main(%arg0: !spirv.arm.tensor<1x16xf32>, %arg1: !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32> {
    // CHECK: %[[NAME:.*]] = spirv.Constant [83 : i8, 101 : i8, 108 : i8, 102 : i8, 65 : i8, 116 : i8, 116 : i8, 101 : i8, 110 : i8, 116 : i8, 105 : i8, 111 : i8, 110 : i8, 79 : i8, 112 : i8] : !spirv.array<15 x i8>
    %name = spirv.Constant dense<[83, 101, 108, 102, 65, 116, 116, 101, 110, 116, 105, 111, 110, 79, 112]> : tensor<15xi8> : !spirv.array<15 x i8>
    // CHECK: %[[ATTRS:.*]] = spirv.Constant [123 : i8, 34 : i8, 111 : i8, 112 : i8, 101 : i8, 114 : i8, 97 : i8, 116 : i8, 111 : i8, 114 : i8, 95 : i8, 110 : i8, 97 : i8, 109 : i8, 101 : i8, 34 : i8, 58 : i8, 34 : i8, 83 : i8, 101 : i8, 108 : i8, 102 : i8, 65 : i8, 116 : i8, 116 : i8, 101 : i8, 110 : i8, 116 : i8, 105 : i8, 111 : i8, 110 : i8, 79 : i8, 112 : i8, 34 : i8, 125 : i8] : !spirv.array<35 x i8>
    %attrs = spirv.Constant dense<[123, 34, 111, 112, 101, 114, 97, 116, 111, 114, 95, 110, 97, 109, 101, 34, 58, 34, 83, 101, 108, 102, 65, 116, 116, 101, 110, 116, 105, 111, 110, 79, 112, 34, 125]> : tensor<35xi8> : !spirv.array<35 x i8>
    // CHECK: %[[CALL:.*]] = spirv.ExperimentalML.Call opcode = 0, %[[NAME]], %[[ATTRS]], %arg0, %arg1 : (!spirv.array<15 x i8>, !spirv.array<35 x i8>, !spirv.arm.tensor<1x16xf32>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
    %0 = spirv.ExperimentalML.Call opcode = 0, %name, %attrs, %arg0, %arg1 : (!spirv.array<15 x i8>, !spirv.array<35 x i8>, !spirv.arm.tensor<1x16xf32>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
    // CHECK: spirv.ARM.GraphOutputs %[[CALL]] : !spirv.arm.tensor<1x16xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x16xf32>
  }
}
