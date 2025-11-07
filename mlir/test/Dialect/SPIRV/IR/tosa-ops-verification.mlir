// RUN: mlir-opt --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// invalid reshape op
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  spirv.ARM.Graph @reshape(%arg0: !spirv.arm.tensor<16x16xi32>, %arg1: !spirv.arm.tensor<2xi32>) -> (!spirv.arm.tensor<16x4xi32>) {
    // expected-error @+1 {{Cannot reshape 256 elements into 64}}
    %1 = spirv.Tosa.Reshape  %arg0, %arg1: !spirv.arm.tensor<16x16xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<16x4xi32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<16x4xi32>
  }
}

//===----------------------------------------------------------------------===//
// invalid tosaavgpool2dop
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  spirv.ARM.Graph @avgpool2d(%arg0: !spirv.arm.tensor<1x11x44x3xi8>, %arg1: !spirv.arm.tensor<1xi8>, %arg2: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1x9x42x3xi16>) {
    // expected-error @+1 {{input and output element types must be the same}}
    %0 = spirv.Tosa.AvgPool2D kernel = dense<3> : !spirv.arm.tensor<2xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>, pad = dense<0> : !spirv.arm.tensor<4xi32>, acc_type = <INT32>, %arg0, %arg1, %arg2 : !spirv.arm.tensor<1x11x44x3xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x42x3xi16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x9x42x3xi16>
  }
}

//===----------------------------------------------------------------------===//
// Invalid Conv2D op: Input and result type mismatch
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  spirv.ARM.Graph @conv2d_quant_mismatch(%arg0: !spirv.arm.tensor<1x16x16x1xf32>, %arg1: !spirv.arm.tensor<8x3x3x1xi8>, %arg2: !spirv.arm.tensor<8xi32>) -> (!spirv.arm.tensor<1x14x14x8xi32>) {
    %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
    %1 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi8>
    // expected-error @+1 {{expect result type to be f32, got 'i32'}}
    %2 = spirv.Tosa.Conv2D pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>, dilation = dense<1> : !spirv.arm.tensor<2xi32>, acc_type = <INT32>, local_bound = true, %arg0, %arg1, %arg2, %0, %1 : !spirv.arm.tensor<1x16x16x1xf32>, !spirv.arm.tensor<8x3x3x1xi8>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x14x14x8xi32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<1x14x14x8xi32>
  }
}

//===----------------------------------------------------------------------===//
// Invalid Conv2D op: zero point set bad with quant
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  spirv.ARM.Graph @conv2d_zero_point_failure(%arg0: !spirv.arm.tensor<1x16x16x1xf32>, %arg1: !spirv.arm.tensor<8x3x3x1xf32>, %arg2: !spirv.arm.tensor<8xf32>) -> (!spirv.arm.tensor<1x14x14x8xf32>) {
    %0 = spirv.Constant dense<1.0> : !spirv.arm.tensor<1xf32>
    %1 = spirv.Constant dense<0.0> : !spirv.arm.tensor<1xf32>
    // expected-error @+1 {{input_zp element value must be zero for non-int8 types}}
    %2 = spirv.Tosa.Conv2D pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>, dilation = dense<1> : !spirv.arm.tensor<2xi32>, acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, %0, %1 : !spirv.arm.tensor<1x16x16x1xf32>, !spirv.arm.tensor<8x3x3x1xf32>, !spirv.arm.tensor<8xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x14x14x8xf32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<1x14x14x8xf32>
  }
}

//===----------------------------------------------------------------------===//
// Invalid MatMul op: Input and result type mismatch
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph]> {
  spirv.ARM.Graph @matmul(%arg0: !spirv.arm.tensor<1x4x4xi16>, %arg1: !spirv.arm.tensor<1x4x4xi16>, %arg2: !spirv.arm.tensor<1xi8>, %arg3: !spirv.arm.tensor<1xi8>) -> (!spirv.arm.tensor<1x4x4xi32>) {
    // expected-error @+1 {{'spirv.Tosa.MatMul' op expect result element type to be i64, got 'i32'}}
    %0 = spirv.Tosa.MatMul %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1x4x4xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x4xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x4x4xi32>
  }
}
