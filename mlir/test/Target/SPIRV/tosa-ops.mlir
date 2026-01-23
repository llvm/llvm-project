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

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @conv2d_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x65535x3x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv2d_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<7x1x1x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv2d_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1xi32>, UniformConstant>
  spirv.GlobalVariable @conv2d_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x65536x2x7xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @conv2d_int, @conv2d_int_arg_0, @conv2d_int_arg_1, @conv2d_int_arg_2, @conv2d_int_res_0
  spirv.ARM.Graph @conv2d_int(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
    %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
    %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
    %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65536x2x7xi32>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @conv2d_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x34x18x27xf16>, UniformConstant>
  spirv.GlobalVariable @conv2d_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<11x1x1x27xf16>, UniformConstant>
  spirv.GlobalVariable @conv2d_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<11xf16>, UniformConstant>
  spirv.GlobalVariable @conv2d_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x34x18x11xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @conv2d_fp, @conv2d_fp_arg_0, @conv2d_fp_arg_1, @conv2d_fp_arg_2, @conv2d_fp_res_0
  spirv.ARM.Graph @conv2d_fp(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
    %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    // CHECK: {{%.*}} = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
    %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x34x18x11xf16>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @conv3d_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x9x21x14x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv3d_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x1x2x1x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv3d_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1xi32>, UniformConstant>
  spirv.GlobalVariable @conv3d_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x9x20x14x2xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @conv3d_int, @conv3d_int_arg_0, @conv3d_int_arg_1, @conv3d_int_arg_2, @conv3d_int_res_0
  spirv.ARM.Graph @conv3d_int(%arg0: !spirv.arm.tensor<1x9x21x14x1xi8>, %arg1: !spirv.arm.tensor<2x1x2x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x9x20x14x2xi32>) {
    %5 = spirv.Constant dense<123> : !spirv.arm.tensor<1xi8>
    %6 = spirv.Constant dense<121> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
    %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x9x20x14x2xi32>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x9x20x14x2xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @conv3d_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x65539x1x2xf32>, UniformConstant>
  spirv.GlobalVariable @conv3d_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x1x1x1x2xf32>, UniformConstant>
  spirv.GlobalVariable @conv3d_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1xf32>, UniformConstant>
  spirv.GlobalVariable @conv3d_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x3x65540x2x1xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @conv3d_fp, @conv3d_fp_arg_0, @conv3d_fp_arg_1, @conv3d_fp_arg_2, @conv3d_fp_res_0
  spirv.ARM.Graph @conv3d_fp(%arg0: !spirv.arm.tensor<1x2x65539x1x2xf32>, %arg1: !spirv.arm.tensor<1x1x1x1x2xf32>, %arg2: !spirv.arm.tensor<1xf32>) -> (!spirv.arm.tensor<1x3x65540x2x1xf32>) {
    %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
    %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
    // CHECK: {{%.*}} = spirv.Tosa.Conv3D pad = [0, 1, 1, 0, 0, 1], stride = [1, 1, 1], dilation = [1, 1, 7], acc_type = <FP32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
    %7 = spirv.Tosa.Conv3D pad = [0, 1, 1, 0, 0, 1], stride = [1, 1, 1], dilation = [1, 1, 7], acc_type = <FP32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x3x65540x2x1xf32>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x3x65540x2x1xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @depthwiseconv2d_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x4x65537x1xi8>, UniformConstant>
  spirv.GlobalVariable @depthwiseconv2d_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x3x1x4xi8>, UniformConstant>
  spirv.GlobalVariable @depthwiseconv2d_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<4xi32>, UniformConstant>
  spirv.GlobalVariable @depthwiseconv2d_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x4x32762x4xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @depthwiseconv2d_int, @depthwiseconv2d_int_arg_0, @depthwiseconv2d_int_arg_1, @depthwiseconv2d_int_arg_2, @depthwiseconv2d_int_res_0
  spirv.ARM.Graph @depthwiseconv2d_int(%arg0: !spirv.arm.tensor<1x4x65537x1xi8>, %arg1: !spirv.arm.tensor<1x3x1x4xi8>, %arg2: !spirv.arm.tensor<4xi32>) -> (!spirv.arm.tensor<1x4x32762x4xi32>) {
    %5 = spirv.Constant dense<58> : !spirv.arm.tensor<1xi8>
    %6 = spirv.Constant dense<-106> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 2], dilation = [7, 7], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
    %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 2], dilation = [7, 7], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x4x32762x4xi32>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x4x32762x4xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @depthwiseconv2d_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x65540x1x3xf32>, UniformConstant>
  spirv.GlobalVariable @depthwiseconv2d_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x1x3x1xf32>, UniformConstant>
  spirv.GlobalVariable @depthwiseconv2d_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1xf32>, UniformConstant>
  spirv.GlobalVariable @depthwiseconv2d_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x65541x2x3xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @depthwiseconv2d_fp, @depthwiseconv2d_fp_arg_0, @depthwiseconv2d_fp_arg_1, @depthwiseconv2d_fp_arg_2, @depthwiseconv2d_fp_res_0
  spirv.ARM.Graph @depthwiseconv2d_fp(%arg0: !spirv.arm.tensor<1x65540x1x3xf32>, %arg1: !spirv.arm.tensor<1x1x3x1xf32>, %arg2: !spirv.arm.tensor<1xf32>) -> (!spirv.arm.tensor<1x65541x2x3xf32>) {
    %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
    %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
    // CHECK: {{%.*}} = spirv.Tosa.DepthwiseConv2D pad = [0, 1, 1, 1], stride = [1, 2], dilation = [1, 7], acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
    %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 1, 1, 1], stride = [1, 2], dilation = [1, 7], acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65541x2x3xf32>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65541x2x3xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @transposeconv2d_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x13x33x3xi16>, UniformConstant>
  spirv.GlobalVariable @transposeconv2d_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<11x1x3x3xi8>, UniformConstant>
  spirv.GlobalVariable @transposeconv2d_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1xi64>, UniformConstant>
  spirv.GlobalVariable @transposeconv2d_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x13x35x11xi64>, UniformConstant>
  spirv.ARM.GraphEntryPoint @transposeconv2d_int, @transposeconv2d_int_arg_0, @transposeconv2d_int_arg_1, @transposeconv2d_int_arg_2, @transposeconv2d_int_res_0
  spirv.ARM.Graph @transposeconv2d_int(%arg0: !spirv.arm.tensor<1x13x33x3xi16>, %arg1: !spirv.arm.tensor<11x1x3x3xi8>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x13x35x11xi64>) {
    %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
    %5 = spirv.Constant dense<88> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
    %6 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x13x35x11xi64>
    spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x13x35x11xi64>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @transposeconv2d_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<10x24x9x13xf16>, UniformConstant>
  spirv.GlobalVariable @transposeconv2d_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<14x1x1x13xf16>, UniformConstant>
  spirv.GlobalVariable @transposeconv2d_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<14xf16>, UniformConstant>
  spirv.GlobalVariable @transposeconv2d_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<10x25x65x14xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @transposeconv2d_fp, @transposeconv2d_fp_arg_0, @transposeconv2d_fp_arg_1, @transposeconv2d_fp_arg_2, @transposeconv2d_fp_res_0
  spirv.ARM.Graph @transposeconv2d_fp(%arg0: !spirv.arm.tensor<10x24x9x13xf16>, %arg1: !spirv.arm.tensor<14x1x1x13xf16>, %arg2: !spirv.arm.tensor<14xf16>) -> (!spirv.arm.tensor<10x25x65x14xf16>) {
    %4 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    // CHECK: {{%.*}} = spirv.Tosa.TransposeConv2D out_pad = [0, 1, 0, 0], stride = [1, 8], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
    %6 = spirv.Tosa.TransposeConv2D out_pad = [0, 1, 0, 0], stride = [1, 8], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<10x25x65x14xf16>
    spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<10x25x65x14xf16>
  }
}
