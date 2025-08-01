// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphConstant
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.0, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  // CHECK: spirv.ARM.GraphConstant {graph_constant_id = 42 : i32} : !spirv.arm.tensor<14xi32>
  %0 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<14xi32>

  // CHECK: spirv.GlobalVariable [[VARARG0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  spirv.GlobalVariable @main_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  // CHECK: spirv.GlobalVariable [[VARRES0:@.*]] bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x3xi16>, UniformConstant>
  spirv.GlobalVariable @main_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x3xi16>, UniformConstant>
  // CHECK: spirv.ARM.GraphEntryPoint [[GN:@.*]], [[VARARG0]], [[VARRES0]]
  spirv.ARM.GraphEntryPoint @main, @main_arg_0, @main_res_0
  // CHECK: spirv.ARM.Graph [[GN]]({{%.*}}: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<2x3xi16> attributes {entry_point = true} {
  spirv.ARM.Graph @main(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<2x3xi16> attributes {entry_point = true} {
    // CHECK: [[CONST2:%.*]] = spirv.ARM.GraphConstant {graph_constant_id = 42 : i32} : !spirv.arm.tensor<2x3xi16>
    %1 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<2x3xi16>
    // CHECK: spirv.ARM.GraphOutputs [[OUT:%.*]] : !spirv.arm.tensor<2x3xi16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<2x3xi16>
  }

  // CHECK: spirv.ARM.Graph {{@.*}}({{%.*}}: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> {
  spirv.ARM.Graph @empty_graph(%arg0: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> {
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x16x16x16xi8>
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<1x16x16x16xi8>
  }
}
