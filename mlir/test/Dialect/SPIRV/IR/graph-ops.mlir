// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph and spirv.ARM.GraphOutputs
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.0, [VulkanMemoryModel, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  // CHECK: spirv.ARM.Graph {{@.*}}({{%.*}}: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
  spirv.ARM.Graph @graphAndOutputs(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<14x19xi16>
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphConstant
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.0, [VulkanMemoryModel, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  // CHECK: spirv.ARM.Graph {{@.*}}() -> !spirv.arm.tensor<2x3xi16> {
  spirv.ARM.Graph @graphConstant() -> !spirv.arm.tensor<2x3xi16> {
    // CHECK: [[CONST:%.*]] = spirv.ARM.GraphConstant {graph_constant_id = 42 : i32} : !spirv.arm.tensor<2x3xi16>
    %0 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<2x3xi16>
    // CHECK: spirv.ARM.GraphOutputs [[CONST:%.*]] : !spirv.arm.tensor<2x3xi16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3xi16>
  }
}
// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphEntryPoint
//===----------------------------------------------------------------------===//


spirv.module Logical Vulkan requires #spirv.vce<v1.0, [VulkanMemoryModel, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  // CHECK: spirv.GlobalVariable [[VARARG0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  spirv.GlobalVariable @entrypoint_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  // CHECK: spirv.GlobalVariable [[VARRES0:@.*]] bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  spirv.GlobalVariable @entrypoint_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
  // CHECK: spirv.ARM.GraphEntryPoint [[GN:@.*]], [[VARARG0]], [[VARRES0]]
  spirv.ARM.GraphEntryPoint @entrypoint, @entrypoint_arg_0, @entrypoint_res_0
  // CHECK: spirv.ARM.Graph [[GN]]({{%.*}}: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
  spirv.ARM.Graph @entrypoint(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<14x19xi16>
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Multiple spirv.ARM.Graphs
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.0, [VulkanMemoryModel, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  // CHECK: spirv.ARM.Graph {{@.*}}({{%.*}}: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
  spirv.ARM.Graph @graph1(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<14x19xi16>
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
  }

  // CHECK: spirv.ARM.Graph {{@.*}}({{%.*}}: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> {
  spirv.ARM.Graph @graph2(%arg0: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> {
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x16x16x16xi8>
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<1x16x16x16xi8>
  }
}
