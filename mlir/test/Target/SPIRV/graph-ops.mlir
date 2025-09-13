// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv %s | spirv-val %}

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  // CHECK: spirv.GlobalVariable [[VARARG0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  spirv.GlobalVariable @main_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  // CHECK: spirv.GlobalVariable [[VARRES0:@.*]] bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x3xi16>, UniformConstant>
  spirv.GlobalVariable @main_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x3xi16>, UniformConstant>
  // CHECK: spirv.ARM.GraphEntryPoint [[GN:@.*]], [[VARARG0]], [[VARRES0]]
  spirv.ARM.GraphEntryPoint @main, @main_arg_0, @main_res_0
  // CHECK: spirv.ARM.Graph [[GN]]({{%.*}}: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<2x3xi16> attributes {entry_point = true} {
  spirv.ARM.Graph @main(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<2x3xi16> attributes {entry_point = true} {
    // CHECK: [[CONST2:%.*]] = spirv.ARM.GraphConstant {graph_constant_id = 42 : i32} : !spirv.arm.tensor<2x3xi16>
    %0 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<2x3xi16>
    // CHECK: spirv.ARM.GraphOutputs [[OUT:%.*]] : !spirv.arm.tensor<2x3xi16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3xi16>
  }

  // CHECK: spirv.ARM.Graph {{@.*}}({{%.*}}: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> attributes {entry_point = false} {
  spirv.ARM.Graph @empty_graph(%arg0: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> {
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x16x16x16xi8>
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<1x16x16x16xi8>
  }
}
