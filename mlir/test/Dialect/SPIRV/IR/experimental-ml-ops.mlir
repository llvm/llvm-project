// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ExperimentalML.Call
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @experimental_ml_call(%arg0: !spirv.arm.tensor<1x16xf32>, %arg1: !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32> {
  // CHECK: %[[NAME:.*]] = spirv.Constant dense<[83, 101, 108, 102, 65, 116, 116, 101, 110, 116, 105, 111, 110, 79, 112]> : tensor<15xi8> : !spirv.array<15 x i8>
  %name = spirv.Constant dense<[83, 101, 108, 102, 65, 116, 116, 101, 110, 116, 105, 111, 110, 79, 112]> : tensor<15xi8> : !spirv.array<15 x i8>
  // CHECK: {{%.*}} = spirv.ExperimentalML.Call opcode = 0, %[[NAME]], %arg0, %arg1 : (!spirv.array<15 x i8>, !spirv.arm.tensor<1x16xf32>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
  %0 = spirv.ExperimentalML.Call opcode = 0, %name, %arg0, %arg1 : (!spirv.array<15 x i8>, !spirv.arm.tensor<1x16xf32>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x16xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x16xf32>
}
