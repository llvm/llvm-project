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
    // CHECK: {{%.*}} = spirv.Tosa.ArgMax %arg0 {axis = 3 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
    %2 = spirv.Tosa.ArgMax %arg0 {axis = 3 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
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
    // CHECK: {{%.*}} = spirv.Tosa.ArgMax %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
    %2 = spirv.Tosa.ArgMax %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x2x14xi32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x2x14xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.AvgPool2D - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @avgpool2d_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x3x65537x1xi8>, UniformConstant>
  spirv.GlobalVariable @avgpool2d_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x32768x1xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @avgpool2d_int, @avgpool2d_int_arg_0, @avgpool2d_int_res_0
  spirv.ARM.Graph @avgpool2d_int(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi8>) {
    %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi8>
    %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.AvgPool2D %arg0, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<INT32>, kernel = dense<3> : !spirv.arm.tensor<2xi32>, pad = dense<[0, 1, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
    %6 = spirv.Tosa.AvgPool2D %arg0, %4, %5 {acc_type = #spirv.tosa_ext_acc_type<INT32>, kernel = dense<3> : !spirv.arm.tensor<2xi32>, pad = dense<[0, 1, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x32768x1xi8>
    spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.AvgPool2D - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @avgpool2d_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x65533x2xf32>, UniformConstant>
  spirv.GlobalVariable @avgpool2d_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x65532x2xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @avgpool2d_fp, @avgpool2d_fp_arg_0, @avgpool2d_fp_res_0
  spirv.ARM.Graph @avgpool2d_fp(%arg0: !spirv.arm.tensor<1x2x65533x2xf32>) -> (!spirv.arm.tensor<1x2x65532x2xf32>) {
    %4 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
    %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
    // CHECK: {{%.*}} = spirv.Tosa.AvgPool2D %arg0, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<FP32>, kernel = dense<2> : !spirv.arm.tensor<2xi32>, pad = dense<[1, 0, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
    %6 = spirv.Tosa.AvgPool2D %arg0, %4, %5 {acc_type = #spirv.tosa_ext_acc_type<FP32>, kernel = dense<2> : !spirv.arm.tensor<2xi32>, pad = dense<[1, 0, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x65532x2xf32>
    spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x65532x2xf32>
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
    // CHECK: {{%.*}} = spirv.Tosa.Conv2D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<INT32>, dilation = dense<[7, 1]> : !spirv.arm.tensor<2xi32>, local_bound = false, pad = dense<[1, 0, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
    %7 = spirv.Tosa.Conv2D %arg0, %arg1, %arg2, %5, %6 {acc_type = #spirv.tosa_ext_acc_type<INT32>, dilation = dense<[7, 1]> : !spirv.arm.tensor<2xi32>, local_bound = false, pad = dense<[1, 0, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
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
    // CHECK: {{%.*}} = spirv.Tosa.Conv2D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<FP16>, dilation = dense<1> : !spirv.arm.tensor<2xi32>, local_bound = true, pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
    %7 = spirv.Tosa.Conv2D %arg0, %arg1, %arg2, %5, %6 {acc_type = #spirv.tosa_ext_acc_type<FP16>, dilation = dense<1> : !spirv.arm.tensor<2xi32>, local_bound = true, pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
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
    // CHECK: {{%.*}} = spirv.Tosa.Conv3D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<INT32>, dilation = dense<1> : !spirv.arm.tensor<3xi32>, local_bound = false, pad = dense<0> : !spirv.arm.tensor<6xi32>, stride = dense<1> : !spirv.arm.tensor<3xi32>} : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
    %7 = spirv.Tosa.Conv3D %arg0, %arg1, %arg2, %5, %6 {acc_type = #spirv.tosa_ext_acc_type<INT32>, dilation = dense<1> : !spirv.arm.tensor<3xi32>, local_bound = false, pad = dense<0> : !spirv.arm.tensor<6xi32>, stride = dense<1> : !spirv.arm.tensor<3xi32>} : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
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
    // CHECK: {{%.*}} = spirv.Tosa.Conv3D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<FP32>, dilation = dense<[1, 1, 7]> : !spirv.arm.tensor<3xi32>, local_bound = false, pad = dense<[0, 1, 1, 0, 0, 1]> : !spirv.arm.tensor<6xi32>, stride = dense<1> : !spirv.arm.tensor<3xi32>} : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
    %7 = spirv.Tosa.Conv3D %arg0, %arg1, %arg2, %5, %6 {acc_type = #spirv.tosa_ext_acc_type<FP32>, dilation = dense<[1, 1, 7]> : !spirv.arm.tensor<3xi32>, local_bound = false, pad = dense<[0, 1, 1, 0, 0, 1]> : !spirv.arm.tensor<6xi32>, stride = dense<1> : !spirv.arm.tensor<3xi32>} : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
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
    // CHECK: {{%.*}} = spirv.Tosa.DepthwiseConv2D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<INT32>, dilation = dense<7> : !spirv.arm.tensor<2xi32>, local_bound = false, pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
    %7 = spirv.Tosa.DepthwiseConv2D %arg0, %arg1, %arg2, %5, %6 {acc_type = #spirv.tosa_ext_acc_type<INT32>, dilation = dense<7> : !spirv.arm.tensor<2xi32>, local_bound = false, pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
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
    // CHECK: {{%.*}} = spirv.Tosa.DepthwiseConv2D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<FP32>, dilation = dense<[1, 7]> : !spirv.arm.tensor<2xi32>, local_bound = true, pad = dense<[0, 1, 1, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
    %7 = spirv.Tosa.DepthwiseConv2D %arg0, %arg1, %arg2, %5, %6 {acc_type = #spirv.tosa_ext_acc_type<FP32>, dilation = dense<[1, 7]> : !spirv.arm.tensor<2xi32>, local_bound = true, pad = dense<[0, 1, 1, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65541x2x3xf32>
    spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65541x2x3xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.FFT2D - EXT-FFT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @fft2d_fft_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x32x32xf32>, UniformConstant>
  spirv.GlobalVariable @fft2d_fft_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x32x32xf32>, UniformConstant>
  spirv.GlobalVariable @fft2d_fft_res_0 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1x32x32xf32>, UniformConstant>
  spirv.GlobalVariable @fft2d_fft_res_1 bind(0, 3) : !spirv.ptr<!spirv.arm.tensor<1x32x32xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @fft2d_fft, @fft2d_fft_arg_0, @fft2d_fft_arg_1, @fft2d_fft_res_0, @fft2d_fft_res_1
  spirv.ARM.Graph @fft2d_fft(%arg0: !spirv.arm.tensor<1x32x32xf32>, %arg1: !spirv.arm.tensor<1x32x32xf32>) -> (!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.FFT2D %arg0, %arg1 {inverse = true, local_bound = false} : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    %out = spirv.Tosa.FFT2D %arg0, %arg1 {inverse = true, local_bound = false} : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    %out0 = spirv.CompositeExtract %out[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32] :  !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    %out1 = spirv.CompositeExtract %out[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>
    spirv.ARM.GraphOutputs  %out0, %out1 : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.MatMul - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @matmul_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<8x2x3xi8>, UniformConstant>
  spirv.GlobalVariable @matmul_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<8x3x8xi8>, UniformConstant>
  spirv.GlobalVariable @matmul_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<8x2x8xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @matmul_int, @matmul_int_arg_0, @matmul_int_arg_1, @matmul_int_res_0
  spirv.ARM.Graph @matmul_int(%arg0: !spirv.arm.tensor<8x2x3xi8>, %arg1: !spirv.arm.tensor<8x3x8xi8>) -> (!spirv.arm.tensor<8x2x8xi32>) {
    %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
    %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.MatMul %arg0, %arg1, {{%.*}}, {{%.*}} : !spirv.arm.tensor<8x2x3xi8>, !spirv.arm.tensor<8x3x8xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<8x2x8xi32>
    %2 = spirv.Tosa.MatMul %arg0, %arg1, %0, %1 : !spirv.arm.tensor<8x2x3xi8>, !spirv.arm.tensor<8x3x8xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<8x2x8xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<8x2x8xi32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<8x2x8xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.MatMul - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @matmul_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<15x39x50xf16>, UniformConstant>
  spirv.GlobalVariable @matmul_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<15x50x24xf16>, UniformConstant>
  spirv.GlobalVariable @matmul_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<15x39x24xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @matmul_fp, @matmul_fp_arg_0, @matmul_fp_arg_1, @matmul_fp_res_0
  spirv.ARM.Graph @matmul_fp(%arg0: !spirv.arm.tensor<15x39x50xf16>, %arg1: !spirv.arm.tensor<15x50x24xf16>) -> (!spirv.arm.tensor<15x39x24xf16>) {
    %0 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    %1 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    // CHECK: {{%.*}} = spirv.Tosa.MatMul %arg0, %arg1, {{%.*}}, {{%.*}} : !spirv.arm.tensor<15x39x50xf16>, !spirv.arm.tensor<15x50x24xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<15x39x24xf16>
    %2 = spirv.Tosa.MatMul %arg0, %arg1, %0, %1 : !spirv.arm.tensor<15x39x50xf16>, !spirv.arm.tensor<15x50x24xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<15x39x24xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<15x39x24xf16>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<15x39x24xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.MaxPool2D - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @maxpool2d_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x3x65537x1xi8>, UniformConstant>
  spirv.GlobalVariable @maxpool2d_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x32769x1xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @maxpool2d_int, @maxpool2d_int_arg_0, @maxpool2d_int_res_0
  spirv.ARM.Graph @maxpool2d_int(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32769x1xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.MaxPool2D %arg0 {kernel = dense<[3, 2]> : !spirv.arm.tensor<2xi32>, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>, pad = dense<[1, 0, 0, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi8>
    %4 = spirv.Tosa.MaxPool2D %arg0 {kernel = dense<[3, 2]> : !spirv.arm.tensor<2xi32>, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>, pad = dense<[1, 0, 0, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 2]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x32769x1xi8>
    spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x2x32769x1xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.MaxPool2D - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @maxpool2d_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x6x65536x1xf32>, UniformConstant>
  spirv.GlobalVariable @maxpool2d_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x3x32769x1xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @maxpool2d_fp, @maxpool2d_fp_arg_0, @maxpool2d_fp_res_0
  spirv.ARM.Graph @maxpool2d_fp(%arg0: !spirv.arm.tensor<1x6x65536x1xf32>) -> (!spirv.arm.tensor<1x3x32769x1xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.MaxPool2D %arg0 {kernel = dense<[3, 2]> : !spirv.arm.tensor<2xi32>, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>, pad = dense<[1, 0, 1, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<2> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x6x65536x1xf32> -> !spirv.arm.tensor<1x3x32769x1xf32>
    %4 = spirv.Tosa.MaxPool2D %arg0 {kernel = dense<[3, 2]> : !spirv.arm.tensor<2xi32>, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>, pad = dense<[1, 0, 1, 1]> : !spirv.arm.tensor<4xi32>, stride = dense<2> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x6x65536x1xf32> -> !spirv.arm.tensor<1x3x32769x1xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x3x32769x1xf32>
    spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x3x32769x1xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.RFFT2D - EXT-FFT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @rfft2d_fft_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x32x32xf32>, UniformConstant>
  spirv.GlobalVariable @rfft2d_fft_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x32x17xf32>, UniformConstant>
  spirv.GlobalVariable @rfft2d_fft_res_1 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<1x32x17xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @rfft2d_fft, @rfft2d_fft_arg_0, @rfft2d_fft_res_0, @rfft2d_fft_res_1
  spirv.ARM.Graph @rfft2d_fft(%arg0: !spirv.arm.tensor<1x32x32xf32>) -> (!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>) {
    %0 = spirv.Constant false
    // CHECK: {{%.*}} = spirv.Tosa.RFFT2D %arg0 {local_bound = false} : !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    %out = spirv.Tosa.RFFT2D %arg0 {local_bound = false} : !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    %out0 = spirv.CompositeExtract %out[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    %out1 = spirv.CompositeExtract %out[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>
    spirv.ARM.GraphOutputs %out0, %out1 : !spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>
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
    // CHECK: {{%.*}} = spirv.Tosa.TransposeConv2D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<INT48>, local_bound = false, out_pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
    %6 = spirv.Tosa.TransposeConv2D %arg0, %arg1, %arg2, %4, %5 {acc_type = #spirv.tosa_ext_acc_type<INT48>, local_bound = false, out_pad = dense<0> : !spirv.arm.tensor<4xi32>, stride = dense<1> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
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
    // CHECK: {{%.*}} = spirv.Tosa.TransposeConv2D %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} {acc_type = #spirv.tosa_ext_acc_type<FP16>, local_bound = true, out_pad = dense<[0, 1, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 8]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
    %6 = spirv.Tosa.TransposeConv2D %arg0, %arg1, %arg2, %4, %5 {acc_type = #spirv.tosa_ext_acc_type<FP16>, local_bound = true, out_pad = dense<[0, 1, 0, 0]> : !spirv.arm.tensor<4xi32>, stride = dense<[1, 8]> : !spirv.arm.tensor<2xi32>} : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<10x25x65x14xf16>
    spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<10x25x65x14xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clamp - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @clamp_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<27x44x55xi8>, UniformConstant>
  spirv.GlobalVariable @clamp_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<27x44x55xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @clamp_int, @clamp_int_arg_0, @clamp_int_res_0
  spirv.ARM.Graph @clamp_int(%arg0: !spirv.arm.tensor<27x44x55xi8>) -> (!spirv.arm.tensor<27x44x55xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.Clamp %arg0 {max_val = -100 : i8, min_val = -102 : i8, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
    %3 = spirv.Tosa.Clamp %arg0 {max_val = -100 : i8, min_val = -102 : i8, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<27x44x55xi8>
    spirv.ARM.GraphOutputs %3 : !spirv.arm.tensor<27x44x55xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clamp - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @clamp_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<18x5x17x6xf32>, UniformConstant>
  spirv.GlobalVariable @clamp_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<18x5x17x6xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @clamp_fp, @clamp_fp_arg_0, @clamp_fp_res_0
  spirv.ARM.Graph @clamp_fp(%arg0: !spirv.arm.tensor<18x5x17x6xf32>) -> (!spirv.arm.tensor<18x5x17x6xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Clamp %arg0 {max_val = 2.38255944E+38 : f32, min_val = -1.19339396E+38 : f32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Ignore>} : !spirv.arm.tensor<18x5x17x6xf32> -> !spirv.arm.tensor<18x5x17x6xf32>
    %3 = spirv.Tosa.Clamp %arg0 {max_val = 2.38255944E+38 : f32, min_val = -1.19339396E+38 : f32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Ignore>} : !spirv.arm.tensor<18x5x17x6xf32> -> !spirv.arm.tensor<18x5x17x6xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<18x5x17x6xf32>
    spirv.ARM.GraphOutputs %3 : !spirv.arm.tensor<18x5x17x6xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Erf - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @erf_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<47x38x51xf32>, UniformConstant>
  spirv.GlobalVariable @erf_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<47x38x51xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @erf_fp, @erf_fp_arg_0, @erf_fp_res_0
  spirv.ARM.Graph @erf_fp(%arg0: !spirv.arm.tensor<47x38x51xf32>) -> (!spirv.arm.tensor<47x38x51xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
    %0 = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<47x38x51xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<47x38x51xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sigmoid - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @sigmoid_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<28x43x45xf32>, UniformConstant>
  spirv.GlobalVariable @sigmoid_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<28x43x45xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @sigmoid_fp, @sigmoid_fp_arg_0, @sigmoid_fp_res_0
  spirv.ARM.Graph @sigmoid_fp(%arg0: !spirv.arm.tensor<28x43x45xf32>) -> (!spirv.arm.tensor<28x43x45xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
    %0 = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<28x43x45xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<28x43x45xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tanh - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @tanh_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<46x50x36xf16>, UniformConstant>
  spirv.GlobalVariable @tanh_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<46x50x36xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @tanh_fp, @tanh_fp_arg_0, @tanh_fp_res_0
  spirv.ARM.Graph @tanh_fp(%arg0: !spirv.arm.tensor<46x50x36xf16>) -> (!spirv.arm.tensor<46x50x36xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
    %0 = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<46x50x36xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x50x36xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @add_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<4x7x3x10xi32>, UniformConstant>
  spirv.GlobalVariable @add_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<4x7x3x1xi32>, UniformConstant>
  spirv.GlobalVariable @add_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<4x7x3x10xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @add_int, @add_int_arg_0, @add_int_arg_1, @add_int_res_0
  spirv.ARM.Graph @add_int(%arg0: !spirv.arm.tensor<4x7x3x10xi32>, %arg1: !spirv.arm.tensor<4x7x3x1xi32>) -> (!spirv.arm.tensor<4x7x3x10xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
    %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x7x3x10xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x7x3x10xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @add_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<26x37x18xf16>, UniformConstant>
  spirv.GlobalVariable @add_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x37x18xf16>, UniformConstant>
  spirv.GlobalVariable @add_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<26x37x18xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @add_fp, @add_fp_arg_0, @add_fp_arg_1, @add_fp_res_0
  spirv.ARM.Graph @add_fp(%arg0: !spirv.arm.tensor<26x37x18xf16>, %arg1: !spirv.arm.tensor<1x37x18xf16>) -> (!spirv.arm.tensor<26x37x18xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<26x37x18xf16>, !spirv.arm.tensor<1x37x18xf16> -> !spirv.arm.tensor<26x37x18xf16>
    %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<26x37x18xf16>, !spirv.arm.tensor<1x37x18xf16> -> !spirv.arm.tensor<26x37x18xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<26x37x18xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<26x37x18xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArithmeticRightShift - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @arithmeticrightshift_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x47x22xi16>, UniformConstant>
  spirv.GlobalVariable @arithmeticrightshift_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<49x47x22xi16>, UniformConstant>
  spirv.GlobalVariable @arithmeticrightshift_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<49x47x22xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @arithmeticrightshift_int, @arithmeticrightshift_int_arg_0, @arithmeticrightshift_int_arg_1, @arithmeticrightshift_int_res_0
  spirv.ARM.Graph @arithmeticrightshift_int(%arg0: !spirv.arm.tensor<1x47x22xi16>, %arg1: !spirv.arm.tensor<49x47x22xi16>) -> (!spirv.arm.tensor<49x47x22xi16>) {
    // CHECK: {{%.*}} = spirv.Tosa.ArithmeticRightShift %arg0, %arg1 {round = true} : !spirv.arm.tensor<1x47x22xi16>, !spirv.arm.tensor<49x47x22xi16> -> !spirv.arm.tensor<49x47x22xi16>
    %1 = spirv.Tosa.ArithmeticRightShift %arg0, %arg1 {round = true} : !spirv.arm.tensor<1x47x22xi16>, !spirv.arm.tensor<49x47x22xi16> -> !spirv.arm.tensor<49x47x22xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<49x47x22xi16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<49x47x22xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseAnd - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @bitwiseand_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<4x1x7x12xi16>, UniformConstant>
  spirv.GlobalVariable @bitwiseand_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<4x13x7x12xi16>, UniformConstant>
  spirv.GlobalVariable @bitwiseand_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<4x13x7x12xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @bitwiseand_int, @bitwiseand_int_arg_0, @bitwiseand_int_arg_1, @bitwiseand_int_res_0
  spirv.ARM.Graph @bitwiseand_int(%arg0: !spirv.arm.tensor<4x1x7x12xi16>, %arg1: !spirv.arm.tensor<4x13x7x12xi16>) -> (!spirv.arm.tensor<4x13x7x12xi16>) {
    // CHECK: {{%.*}} = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
    %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x13x7x12xi16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x13x7x12xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseOr - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @bitwiseor_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<11x30x23xi32>, UniformConstant>
  spirv.GlobalVariable @bitwiseor_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x30x23xi32>, UniformConstant>
  spirv.GlobalVariable @bitwiseor_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<11x30x23xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @bitwiseor_int, @bitwiseor_int_arg_0, @bitwiseor_int_arg_1, @bitwiseor_int_res_0
  spirv.ARM.Graph @bitwiseor_int(%arg0: !spirv.arm.tensor<11x30x23xi32>, %arg1: !spirv.arm.tensor<1x30x23xi32>) -> (!spirv.arm.tensor<11x30x23xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
    %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x30x23xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x30x23xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseXor - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @bitwisexor_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<4x8x13x9xi16>, UniformConstant>
  spirv.GlobalVariable @bitwisexor_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<4x8x1x9xi16>, UniformConstant>
  spirv.GlobalVariable @bitwisexor_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<4x8x13x9xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @bitwisexor_int, @bitwisexor_int_arg_0, @bitwisexor_int_arg_1, @bitwisexor_int_res_0
  spirv.ARM.Graph @bitwisexor_int(%arg0: !spirv.arm.tensor<4x8x13x9xi16>, %arg1: !spirv.arm.tensor<4x8x1x9xi16>) -> (!spirv.arm.tensor<4x8x13x9xi16>) {
    // CHECK: {{%.*}} = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
    %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x8x13x9xi16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x8x13x9xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.IntDiv - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @intdiv_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x65533x1xi32>, UniformConstant>
  spirv.GlobalVariable @intdiv_any_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x65533x1xi32>, UniformConstant>
  spirv.GlobalVariable @intdiv_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x65533x1xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @intdiv_any, @intdiv_any_arg_0, @intdiv_any_arg_1, @intdiv_any_res_0
  spirv.ARM.Graph @intdiv_any(%arg0: !spirv.arm.tensor<1x65533x1xi32>, %arg1: !spirv.arm.tensor<2x65533x1xi32>) -> (!spirv.arm.tensor<2x65533x1xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
    %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x65533x1xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x65533x1xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalAnd - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @logicaland_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x1x7x11xi1>, UniformConstant>
  spirv.GlobalVariable @logicaland_any_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x4x7x11xi1>, UniformConstant>
  spirv.GlobalVariable @logicaland_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x4x7x11xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @logicaland_any, @logicaland_any_arg_0, @logicaland_any_arg_1, @logicaland_any_res_0
  spirv.ARM.Graph @logicaland_any(%arg0: !spirv.arm.tensor<2x1x7x11xi1>, %arg1: !spirv.arm.tensor<2x4x7x11xi1>) -> (!spirv.arm.tensor<2x4x7x11xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
    %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x4x7x11xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x4x7x11xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalLeftShift - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @logicalleftshift_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<7x1x11x4xi8>, UniformConstant>
  spirv.GlobalVariable @logicalleftshift_any_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<7x8x11x4xi8>, UniformConstant>
  spirv.GlobalVariable @logicalleftshift_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<7x8x11x4xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @logicalleftshift_any, @logicalleftshift_any_arg_0, @logicalleftshift_any_arg_1, @logicalleftshift_any_res_0
  spirv.ARM.Graph @logicalleftshift_any(%arg0: !spirv.arm.tensor<7x1x11x4xi8>, %arg1: !spirv.arm.tensor<7x8x11x4xi8>) -> (!spirv.arm.tensor<7x8x11x4xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
    %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<7x8x11x4xi8>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<7x8x11x4xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalRightShift - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @logicalrightshift_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<6x13x1x19xi8>, UniformConstant>
  spirv.GlobalVariable @logicalrightshift_any_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<6x13x6x19xi8>, UniformConstant>
  spirv.GlobalVariable @logicalrightshift_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<6x13x6x19xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @logicalrightshift_any, @logicalrightshift_any_arg_0, @logicalrightshift_any_arg_1, @logicalrightshift_any_res_0
  spirv.ARM.Graph @logicalrightshift_any(%arg0: !spirv.arm.tensor<6x13x1x19xi8>, %arg1: !spirv.arm.tensor<6x13x6x19xi8>) -> (!spirv.arm.tensor<6x13x6x19xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
    %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x13x6x19xi8>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x13x6x19xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalOr - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @logicalor_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<3x6x12x5xi1>, UniformConstant>
  spirv.GlobalVariable @logicalor_any_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<3x6x1x5xi1>, UniformConstant>
  spirv.GlobalVariable @logicalor_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<3x6x12x5xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @logicalor_any, @logicalor_any_arg_0, @logicalor_any_arg_1, @logicalor_any_res_0
  spirv.ARM.Graph @logicalor_any(%arg0: !spirv.arm.tensor<3x6x12x5xi1>, %arg1: !spirv.arm.tensor<3x6x1x5xi1>) -> (!spirv.arm.tensor<3x6x12x5xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
    %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x6x12x5xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x6x12x5xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalXor - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @logicalxor_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<11x4x9x12xi1>, UniformConstant>
  spirv.GlobalVariable @logicalxor_any_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<11x4x9x1xi1>, UniformConstant>
  spirv.GlobalVariable @logicalxor_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<11x4x9x12xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @logicalxor_any, @logicalxor_any_arg_0, @logicalxor_any_arg_1, @logicalxor_any_res_0
  spirv.ARM.Graph @logicalxor_any(%arg0: !spirv.arm.tensor<11x4x9x12xi1>, %arg1: !spirv.arm.tensor<11x4x9x1xi1>) -> (!spirv.arm.tensor<11x4x9x12xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
    %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x4x9x12xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x4x9x12xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @maximum_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x65533x1xi32>, UniformConstant>
  spirv.GlobalVariable @maximum_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x2x65533x2xi32>, UniformConstant>
  spirv.GlobalVariable @maximum_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x65533x2xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @maximum_int, @maximum_int_arg_0, @maximum_int_arg_1, @maximum_int_res_0
  spirv.ARM.Graph @maximum_int(%arg0: !spirv.arm.tensor<1x2x65533x1xi32>, %arg1: !spirv.arm.tensor<1x2x65533x2xi32>) -> (!spirv.arm.tensor<1x2x65533x2xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Maximum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
    %1 = spirv.Tosa.Maximum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x65533x2xi32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<1x2x65533x2xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @maximum_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x12x14x7xf16>, UniformConstant>
  spirv.GlobalVariable @maximum_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<11x12x14x7xf16>, UniformConstant>
  spirv.GlobalVariable @maximum_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<11x12x14x7xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @maximum_fp, @maximum_fp_arg_0, @maximum_fp_arg_1, @maximum_fp_res_0
  spirv.ARM.Graph @maximum_fp(%arg0: !spirv.arm.tensor<1x12x14x7xf16>, %arg1: !spirv.arm.tensor<11x12x14x7xf16>) -> (!spirv.arm.tensor<11x12x14x7xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Maximum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Ignore>} : !spirv.arm.tensor<1x12x14x7xf16>, !spirv.arm.tensor<11x12x14x7xf16> -> !spirv.arm.tensor<11x12x14x7xf16>
    %1 = spirv.Tosa.Maximum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Ignore>} : !spirv.arm.tensor<1x12x14x7xf16>, !spirv.arm.tensor<11x12x14x7xf16> -> !spirv.arm.tensor<11x12x14x7xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x12x14x7xf16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<11x12x14x7xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @minimum_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<15x2x10x11xi32>, UniformConstant>
  spirv.GlobalVariable @minimum_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<15x1x10x11xi32>, UniformConstant>
  spirv.GlobalVariable @minimum_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<15x2x10x11xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @minimum_int, @minimum_int_arg_0, @minimum_int_arg_1, @minimum_int_res_0
  spirv.ARM.Graph @minimum_int(%arg0: !spirv.arm.tensor<15x2x10x11xi32>, %arg1: !spirv.arm.tensor<15x1x10x11xi32>) -> (!spirv.arm.tensor<15x2x10x11xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Minimum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
    %1 = spirv.Tosa.Minimum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<15x2x10x11xi32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<15x2x10x11xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @minimum_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x65531x2x1xf32>, UniformConstant>
  spirv.GlobalVariable @minimum_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x1x2x1xf32>, UniformConstant>
  spirv.GlobalVariable @minimum_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x65531x2x1xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @minimum_fp, @minimum_fp_arg_0, @minimum_fp_arg_1, @minimum_fp_res_0
  spirv.ARM.Graph @minimum_fp(%arg0: !spirv.arm.tensor<1x65531x2x1xf32>, %arg1: !spirv.arm.tensor<1x1x2x1xf32>) -> (!spirv.arm.tensor<1x65531x2x1xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Minimum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<1x65531x2x1xf32>, !spirv.arm.tensor<1x1x2x1xf32> -> !spirv.arm.tensor<1x65531x2x1xf32>
    %1 = spirv.Tosa.Minimum %arg0, %arg1 {nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<1x65531x2x1xf32>, !spirv.arm.tensor<1x1x2x1xf32> -> !spirv.arm.tensor<1x65531x2x1xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65531x2x1xf32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<1x65531x2x1xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Mul - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @mul_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<34x21x39xi32>, UniformConstant>
  spirv.GlobalVariable @mul_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<34x21x1xi32>, UniformConstant>
  spirv.GlobalVariable @mul_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<34x21x39xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @mul_int, @mul_int_arg_0, @mul_int_arg_1, @mul_int_res_0
  spirv.ARM.Graph @mul_int(%arg0: !spirv.arm.tensor<34x21x39xi32>, %arg1: !spirv.arm.tensor<34x21x1xi32>) -> (!spirv.arm.tensor<34x21x39xi32>) {
    %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Mul %arg0, %arg1, {{%.*}} : !spirv.arm.tensor<34x21x39xi32>, !spirv.arm.tensor<34x21x1xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
    %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi32>, !spirv.arm.tensor<34x21x1xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<34x21x39xi32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x39xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Mul - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @mul_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<57x1x55xf16>, UniformConstant>
  spirv.GlobalVariable @mul_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<57x37x55xf16>, UniformConstant>
  spirv.GlobalVariable @mul_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<57x37x55xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @mul_fp, @mul_fp_arg_0, @mul_fp_arg_1, @mul_fp_res_0
  spirv.ARM.Graph @mul_fp(%arg0: !spirv.arm.tensor<57x1x55xf16>, %arg1: !spirv.arm.tensor<57x37x55xf16>) -> (!spirv.arm.tensor<57x37x55xf16>) {
    %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Mul %arg0, %arg1, {{%.*}} : !spirv.arm.tensor<57x1x55xf16>, !spirv.arm.tensor<57x37x55xf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf16>
    %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<57x1x55xf16>, !spirv.arm.tensor<57x37x55xf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<57x37x55xf16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<57x37x55xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pow - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @pow_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x52x53xf16>, UniformConstant>
  spirv.GlobalVariable @pow_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<44x52x53xf16>, UniformConstant>
  spirv.GlobalVariable @pow_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<44x52x53xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @pow_fp, @pow_fp_arg_0, @pow_fp_arg_1, @pow_fp_res_0
  spirv.ARM.Graph @pow_fp(%arg0: !spirv.arm.tensor<1x52x53xf16>, %arg1: !spirv.arm.tensor<44x52x53xf16>) -> (!spirv.arm.tensor<44x52x53xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
    %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<44x52x53xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x52x53xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @sub_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<6x10x6x6xi32>, UniformConstant>
  spirv.GlobalVariable @sub_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x10x6x6xi32>, UniformConstant>
  spirv.GlobalVariable @sub_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<6x10x6x6xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @sub_int, @sub_int_arg_0, @sub_int_arg_1, @sub_int_res_0
  spirv.ARM.Graph @sub_int(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
    %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x10x6x6xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @sub_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x10x13x12xf16>, UniformConstant>
  spirv.GlobalVariable @sub_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<6x10x13x12xf16>, UniformConstant>
  spirv.GlobalVariable @sub_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<6x10x13x12xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @sub_fp, @sub_fp_arg_0, @sub_fp_arg_1, @sub_fp_res_0
  spirv.ARM.Graph @sub_fp(%arg0: !spirv.arm.tensor<1x10x13x12xf16>, %arg1: !spirv.arm.tensor<6x10x13x12xf16>) -> (!spirv.arm.tensor<6x10x13x12xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<1x10x13x12xf16>, !spirv.arm.tensor<6x10x13x12xf16> -> !spirv.arm.tensor<6x10x13x12xf16>
    %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<1x10x13x12xf16>, !spirv.arm.tensor<6x10x13x12xf16> -> !spirv.arm.tensor<6x10x13x12xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x10x13x12xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x13x12xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Table - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @table_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<3x2x15x7xi8>, UniformConstant>
  spirv.GlobalVariable @table_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<3x2x15x7xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @table_int, @table_int_arg_0, @table_int_res_0
  spirv.ARM.Graph @table_int(%arg0: !spirv.arm.tensor<3x2x15x7xi8>) -> (!spirv.arm.tensor<3x2x15x7xi8>) {
    %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<256xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Table %arg0, {{%.*}} : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x7xi8>
    %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x7xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x2x15x7xi8>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @abs_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<5x1x4x4xi32>, UniformConstant>
  spirv.GlobalVariable @abs_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<5x1x4x4xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @abs_int, @abs_int_arg_0, @abs_int_res_0
  spirv.ARM.Graph @abs_int(%arg0: !spirv.arm.tensor<5x1x4x4xi32>) -> (!spirv.arm.tensor<5x1x4x4xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
    %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<5x1x4x4xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<5x1x4x4xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @abs_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<3x6x14x8xf16>, UniformConstant>
  spirv.GlobalVariable @abs_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<3x6x14x8xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @abs_fp, @abs_fp_arg_0, @abs_fp_res_0
  spirv.ARM.Graph @abs_fp(%arg0: !spirv.arm.tensor<3x6x14x8xf16>) -> (!spirv.arm.tensor<3x6x14x8xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x8xf16>
    %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x8xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x6x14x8xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x6x14x8xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseNot - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @bitwisenot_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<12x56x50xi32>, UniformConstant>
  spirv.GlobalVariable @bitwisenot_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<12x56x50xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @bitwisenot_int, @bitwisenot_int_arg_0, @bitwisenot_int_res_0
  spirv.ARM.Graph @bitwisenot_int(%arg0: !spirv.arm.tensor<12x56x50xi32>) -> (!spirv.arm.tensor<12x56x50xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
    %0 = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<12x56x50xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<12x56x50xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Ceil - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @ceil_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<46x55x53xf16>, UniformConstant>
  spirv.GlobalVariable @ceil_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<46x55x53xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @ceil_fp, @ceil_fp_arg_0, @ceil_fp_res_0
  spirv.ARM.Graph @ceil_fp(%arg0: !spirv.arm.tensor<46x55x53xf16>) -> (!spirv.arm.tensor<46x55x53xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
    %0 = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<46x55x53xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x55x53xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clz - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @clz_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x10x7x5xi32>, UniformConstant>
  spirv.GlobalVariable @clz_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<14x10x7x5xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @clz_int, @clz_int_arg_0, @clz_int_res_0
  spirv.ARM.Graph @clz_int(%arg0: !spirv.arm.tensor<14x10x7x5xi32>) -> (!spirv.arm.tensor<14x10x7x5xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
    %0 = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<14x10x7x5xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<14x10x7x5xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cos - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @cos_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<44x49x51xf32>, UniformConstant>
  spirv.GlobalVariable @cos_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<44x49x51xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @cos_fp, @cos_fp_arg_0, @cos_fp_res_0
  spirv.ARM.Graph @cos_fp(%arg0: !spirv.arm.tensor<44x49x51xf32>) -> (!spirv.arm.tensor<44x49x51xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
    %0 = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<44x49x51xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x49x51xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Exp - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @exp_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<37x53x47xf32>, UniformConstant>
  spirv.GlobalVariable @exp_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<37x53x47xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @exp_fp, @exp_fp_arg_0, @exp_fp_res_0
  spirv.ARM.Graph @exp_fp(%arg0: !spirv.arm.tensor<37x53x47xf32>) -> (!spirv.arm.tensor<37x53x47xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
    %0 = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<37x53x47xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<37x53x47xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Floor - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @floor_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<40x52x42xf32>, UniformConstant>
  spirv.GlobalVariable @floor_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<40x52x42xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @floor_fp, @floor_fp_arg_0, @floor_fp_res_0
  spirv.ARM.Graph @floor_fp(%arg0: !spirv.arm.tensor<40x52x42xf32>) -> (!spirv.arm.tensor<40x52x42xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
    %0 = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<40x52x42xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x52x42xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Log - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @log_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<45x43x36xf16>, UniformConstant>
  spirv.GlobalVariable @log_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<45x43x36xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @log_fp, @log_fp_arg_0, @log_fp_res_0
  spirv.ARM.Graph @log_fp(%arg0: !spirv.arm.tensor<45x43x36xf16>) -> (!spirv.arm.tensor<45x43x36xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
    %0 = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<45x43x36xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<45x43x36xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalNot - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @logicalnot_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<54x26x10xi1>, UniformConstant>
  spirv.GlobalVariable @logicalnot_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<54x26x10xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @logicalnot_any, @logicalnot_any_arg_0, @logicalnot_any_res_0
  spirv.ARM.Graph @logicalnot_any(%arg0: !spirv.arm.tensor<54x26x10xi1>) -> (!spirv.arm.tensor<54x26x10xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.LogicalNot %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
    %0 = spirv.Tosa.LogicalNot %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<54x26x10xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<54x26x10xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Negate - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @negate_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<3x1x65540x1xi8>, UniformConstant>
  spirv.GlobalVariable @negate_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<3x1x65540x1xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @negate_int, @negate_int_arg_0, @negate_int_res_0
  spirv.ARM.Graph @negate_int(%arg0: !spirv.arm.tensor<3x1x65540x1xi8>) -> (!spirv.arm.tensor<3x1x65540x1xi8>) {
    %0 = spirv.Constant dense<111> : !spirv.arm.tensor<1xi8>
    %1 = spirv.Constant dense<-32> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Negate %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x1xi8>
    %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x1xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x1x65540x1xi8>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x1x65540x1xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Negate - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @negate_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x7x15x13xf16>, UniformConstant>
  spirv.GlobalVariable @negate_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x7x15x13xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @negate_fp, @negate_fp_arg_0, @negate_fp_res_0
  spirv.ARM.Graph @negate_fp(%arg0: !spirv.arm.tensor<2x7x15x13xf16>) -> (!spirv.arm.tensor<2x7x15x13xf16>) {
    %0 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    %1 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
    // CHECK: {{%.*}} = spirv.Tosa.Negate %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<2x7x15x13xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<2x7x15x13xf16>
    %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<2x7x15x13xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<2x7x15x13xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x7x15x13xf16>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x7x15x13xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reciprocal - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reciprocal_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<38x47x44xf32>, UniformConstant>
  spirv.GlobalVariable @reciprocal_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<38x47x44xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reciprocal_fp, @reciprocal_fp_arg_0, @reciprocal_fp_res_0
  spirv.ARM.Graph @reciprocal_fp(%arg0: !spirv.arm.tensor<38x47x44xf32>) -> (!spirv.arm.tensor<38x47x44xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
    %0 = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<38x47x44xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<38x47x44xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rsqrt - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @rsqrt_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<40x57x56xf32>, UniformConstant>
  spirv.GlobalVariable @rsqrt_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<40x57x56xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @rsqrt_fp, @rsqrt_fp_arg_0, @rsqrt_fp_res_0
  spirv.ARM.Graph @rsqrt_fp(%arg0: !spirv.arm.tensor<40x57x56xf32>) -> (!spirv.arm.tensor<40x57x56xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
    %0 = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<40x57x56xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x57x56xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sin - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @sin_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<49x38x58xf16>, UniformConstant>
  spirv.GlobalVariable @sin_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<49x38x58xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @sin_fp, @sin_fp_arg_0, @sin_fp_res_0
  spirv.ARM.Graph @sin_fp(%arg0: !spirv.arm.tensor<49x38x58xf16>) -> (!spirv.arm.tensor<49x38x58xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
    %0 = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<49x38x58xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<49x38x58xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Select - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @select_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<4x1x4x5xi1>, UniformConstant>
  spirv.GlobalVariable @select_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<4x6x4x5xi8>, UniformConstant>
  spirv.GlobalVariable @select_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<4x6x4x5xi8>, UniformConstant>
  spirv.GlobalVariable @select_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<4x6x4x5xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @select_int, @select_int_arg_0, @select_int_arg_1, @select_int_arg_2, @select_int_res_0
  spirv.ARM.Graph @select_int(%arg0: !spirv.arm.tensor<4x1x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
    %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x6x4x5xi8>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Select - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @select_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<9x2x15x8xi1>, UniformConstant>
  spirv.GlobalVariable @select_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<9x2x15x8xf16>, UniformConstant>
  spirv.GlobalVariable @select_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<9x1x15x8xf16>, UniformConstant>
  spirv.GlobalVariable @select_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<9x2x15x8xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @select_fp, @select_fp_arg_0, @select_fp_arg_1, @select_fp_arg_2, @select_fp_res_0
  spirv.ARM.Graph @select_fp(%arg0: !spirv.arm.tensor<9x2x15x8xi1>, %arg1: !spirv.arm.tensor<9x2x15x8xf16>, %arg2: !spirv.arm.tensor<9x1x15x8xf16>) -> (!spirv.arm.tensor<9x2x15x8xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<9x2x15x8xi1>, !spirv.arm.tensor<9x2x15x8xf16>, !spirv.arm.tensor<9x1x15x8xf16> -> !spirv.arm.tensor<9x2x15x8xf16>
    %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<9x2x15x8xi1>, !spirv.arm.tensor<9x2x15x8xf16>, !spirv.arm.tensor<9x1x15x8xf16> -> !spirv.arm.tensor<9x2x15x8xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<9x2x15x8xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<9x2x15x8xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @equal_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<51x28x59xi32>, UniformConstant>
  spirv.GlobalVariable @equal_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<51x1x59xi32>, UniformConstant>
  spirv.GlobalVariable @equal_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<51x28x59xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @equal_int, @equal_int_arg_0, @equal_int_arg_1, @equal_int_res_0
  spirv.ARM.Graph @equal_int(%arg0: !spirv.arm.tensor<51x28x59xi32>, %arg1: !spirv.arm.tensor<51x1x59xi32>) -> (!spirv.arm.tensor<51x28x59xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
    %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<51x28x59xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<51x28x59xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @equal_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<16x11x5x3xf32>, UniformConstant>
  spirv.GlobalVariable @equal_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<16x1x5x3xf32>, UniformConstant>
  spirv.GlobalVariable @equal_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<16x11x5x3xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @equal_fp, @equal_fp_arg_0, @equal_fp_arg_1, @equal_fp_res_0
  spirv.ARM.Graph @equal_fp(%arg0: !spirv.arm.tensor<16x11x5x3xf32>, %arg1: !spirv.arm.tensor<16x1x5x3xf32>) -> (!spirv.arm.tensor<16x11x5x3xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
    %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<16x11x5x3xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<16x11x5x3xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @greater_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<11x10x10x2xi32>, UniformConstant>
  spirv.GlobalVariable @greater_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<11x10x10x1xi32>, UniformConstant>
  spirv.GlobalVariable @greater_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<11x10x10x2xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @greater_int, @greater_int_arg_0, @greater_int_arg_1, @greater_int_res_0
  spirv.ARM.Graph @greater_int(%arg0: !spirv.arm.tensor<11x10x10x2xi32>, %arg1: !spirv.arm.tensor<11x10x10x1xi32>) -> (!spirv.arm.tensor<11x10x10x2xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
    %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x10x10x2xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x10x10x2xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @greater_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<6x3x12x4xf16>, UniformConstant>
  spirv.GlobalVariable @greater_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<6x3x1x4xf16>, UniformConstant>
  spirv.GlobalVariable @greater_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<6x3x12x4xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @greater_fp, @greater_fp_arg_0, @greater_fp_arg_1, @greater_fp_res_0
  spirv.ARM.Graph @greater_fp(%arg0: !spirv.arm.tensor<6x3x12x4xf16>, %arg1: !spirv.arm.tensor<6x3x1x4xf16>) -> (!spirv.arm.tensor<6x3x12x4xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<6x3x12x4xf16>, !spirv.arm.tensor<6x3x1x4xf16> -> !spirv.arm.tensor<6x3x12x4xi1>
    %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<6x3x12x4xf16>, !spirv.arm.tensor<6x3x1x4xf16> -> !spirv.arm.tensor<6x3x12x4xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x3x12x4xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x3x12x4xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @greaterequal_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<10x17x7x1xi32>, UniformConstant>
  spirv.GlobalVariable @greaterequal_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<10x17x7x16xi32>, UniformConstant>
  spirv.GlobalVariable @greaterequal_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<10x17x7x16xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @greaterequal_int, @greaterequal_int_arg_0, @greaterequal_int_arg_1, @greaterequal_int_res_0
  spirv.ARM.Graph @greaterequal_int(%arg0: !spirv.arm.tensor<10x17x7x1xi32>, %arg1: !spirv.arm.tensor<10x17x7x16xi32>) -> (!spirv.arm.tensor<10x17x7x16xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
    %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<10x17x7x16xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<10x17x7x16xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @greaterequal_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<3x17x6x3xf32>, UniformConstant>
  spirv.GlobalVariable @greaterequal_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x17x6x3xf32>, UniformConstant>
  spirv.GlobalVariable @greaterequal_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<3x17x6x3xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @greaterequal_fp, @greaterequal_fp_arg_0, @greaterequal_fp_arg_1, @greaterequal_fp_res_0
  spirv.ARM.Graph @greaterequal_fp(%arg0: !spirv.arm.tensor<3x17x6x3xf32>, %arg1: !spirv.arm.tensor<1x17x6x3xf32>) -> (!spirv.arm.tensor<3x17x6x3xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<3x17x6x3xf32>, !spirv.arm.tensor<1x17x6x3xf32> -> !spirv.arm.tensor<3x17x6x3xi1>
    %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<3x17x6x3xf32>, !spirv.arm.tensor<1x17x6x3xf32> -> !spirv.arm.tensor<3x17x6x3xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x17x6x3xi1>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x17x6x3xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAll - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reduceall_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<18x22x23x12xi1>, UniformConstant>
  spirv.GlobalVariable @reduceall_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<18x22x1x12xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reduceall_any, @reduceall_any_arg_0, @reduceall_any_res_0
  spirv.ARM.Graph @reduceall_any(%arg0: !spirv.arm.tensor<18x22x23x12xi1>) -> (!spirv.arm.tensor<18x22x1x12xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceAll %arg0 {axis = 2 : i32} : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x1x12xi1>
    %1 = spirv.Tosa.ReduceAll %arg0 {axis = 2 : i32} : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x1x12xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<18x22x1x12xi1>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<18x22x1x12xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAny - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reduceany_any_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<25x13x30x8xi1>, UniformConstant>
  spirv.GlobalVariable @reduceany_any_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<25x13x1x8xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reduceany_any, @reduceany_any_arg_0, @reduceany_any_res_0
  spirv.ARM.Graph @reduceany_any(%arg0: !spirv.arm.tensor<25x13x30x8xi1>) -> (!spirv.arm.tensor<25x13x1x8xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceAny %arg0 {axis = 2 : i32} : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x1x8xi1>
    %1 = spirv.Tosa.ReduceAny %arg0 {axis = 2 : i32} : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x1x8xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<25x13x1x8xi1>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<25x13x1x8xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMax - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reducemax_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<8x30x12x3xi8>, UniformConstant>
  spirv.GlobalVariable @reducemax_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<8x30x1x3xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reducemax_int, @reducemax_int_arg_0, @reducemax_int_res_0
  spirv.ARM.Graph @reducemax_int(%arg0: !spirv.arm.tensor<8x30x12x3xi8>) -> (!spirv.arm.tensor<8x30x1x3xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceMax %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x1x3xi8>
    %2 = spirv.Tosa.ReduceMax %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x1x3xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<8x30x1x3xi8>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<8x30x1x3xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMax - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reducemax_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<16x20x10xf16>, UniformConstant>
  spirv.GlobalVariable @reducemax_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<16x20x1xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reducemax_fp, @reducemax_fp_arg_0, @reducemax_fp_res_0
  spirv.ARM.Graph @reducemax_fp(%arg0: !spirv.arm.tensor<16x20x10xf16>) -> (!spirv.arm.tensor<16x20x1xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceMax %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<16x20x10xf16> -> !spirv.arm.tensor<16x20x1xf16>
    %2 = spirv.Tosa.ReduceMax %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<16x20x10xf16> -> !spirv.arm.tensor<16x20x1xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<16x20x1xf16>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<16x20x1xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMin - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reducemin_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x5x5x1xi8>, UniformConstant>
  spirv.GlobalVariable @reducemin_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x5x1x1xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reducemin_int, @reducemin_int_arg_0, @reducemin_int_res_0
  spirv.ARM.Graph @reducemin_int(%arg0: !spirv.arm.tensor<2x5x5x1xi8>) -> (!spirv.arm.tensor<2x5x1x1xi8>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceMin %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<2x5x5x1xi8> -> !spirv.arm.tensor<2x5x1x1xi8>
    %2 = spirv.Tosa.ReduceMin %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<2x5x5x1xi8> -> !spirv.arm.tensor<2x5x1x1xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x5x1x1xi8>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x5x1x1xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMin - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reducemin_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<27x10x25x9xf16>, UniformConstant>
  spirv.GlobalVariable @reducemin_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<27x10x1x9xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reducemin_fp, @reducemin_fp_arg_0, @reducemin_fp_res_0
  spirv.ARM.Graph @reducemin_fp(%arg0: !spirv.arm.tensor<27x10x25x9xf16>) -> (!spirv.arm.tensor<27x10x1x9xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceMin %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x1x9xf16>
    %2 = spirv.Tosa.ReduceMin %arg0 {axis = 2 : i32, nan_mode = #spirv.tosa_ext_nan_propagation_mode_type<Propagate>} : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x1x9xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<27x10x1x9xf16>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<27x10x1x9xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceProduct - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reduceproduct_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x16x25xf16>, UniformConstant>
  spirv.GlobalVariable @reduceproduct_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x16x1xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reduceproduct_fp, @reduceproduct_fp_arg_0, @reduceproduct_fp_res_0
  spirv.ARM.Graph @reduceproduct_fp(%arg0: !spirv.arm.tensor<2x16x25xf16>) -> (!spirv.arm.tensor<2x16x1xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceProduct %arg0 {axis = 2 : i32} : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x1xf16>
    %1 = spirv.Tosa.ReduceProduct %arg0 {axis = 2 : i32} : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x1xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x16x1xf16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<2x16x1xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceSum - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reducesum_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<20x24x22xi32>, UniformConstant>
  spirv.GlobalVariable @reducesum_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<20x1x22xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reducesum_int, @reducesum_int_arg_0, @reducesum_int_res_0
  spirv.ARM.Graph @reducesum_int(%arg0: !spirv.arm.tensor<20x24x22xi32>) -> (!spirv.arm.tensor<20x1x22xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceSum %arg0 {axis = 1 : i32} : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x1x22xi32>
    %1 = spirv.Tosa.ReduceSum %arg0 {axis = 1 : i32} : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x1x22xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<20x1x22xi32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<20x1x22xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceSum - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reducesum_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<32x32x33xf32>, UniformConstant>
  spirv.GlobalVariable @reducesum_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<32x1x33xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reducesum_fp, @reducesum_fp_arg_0, @reducesum_fp_res_0
  spirv.ARM.Graph @reducesum_fp(%arg0: !spirv.arm.tensor<32x32x33xf32>) -> (!spirv.arm.tensor<32x1x33xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.ReduceSum %arg0 {axis = 1 : i32} : !spirv.arm.tensor<32x32x33xf32> -> !spirv.arm.tensor<32x1x33xf32>
    %1 = spirv.Tosa.ReduceSum %arg0 {axis = 1 : i32} : !spirv.arm.tensor<32x32x33xf32> -> !spirv.arm.tensor<32x1x33xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<32x1x33xf32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<32x1x33xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Concat - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @concat_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<12x13x3x14xi8>, UniformConstant>
  spirv.GlobalVariable @concat_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<12x13x3x14xi8>, UniformConstant>
  spirv.GlobalVariable @concat_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<12x13x3x14xi8>, UniformConstant>
  spirv.GlobalVariable @concat_int_arg_3 bind(0, 3) : !spirv.ptr<!spirv.arm.tensor<12x13x3x14xi8>, UniformConstant>
  spirv.GlobalVariable @concat_int_res_0 bind(0, 4) : !spirv.ptr<!spirv.arm.tensor<12x13x12x14xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @concat_int, @concat_int_arg_0, @concat_int_arg_1, @concat_int_arg_2, @concat_int_arg_3, @concat_int_res_0
  spirv.ARM.Graph @concat_int(%arg0: !spirv.arm.tensor<12x13x3x14xi8>, %arg1: !spirv.arm.tensor<12x13x3x14xi8>, %arg2: !spirv.arm.tensor<12x13x3x14xi8>, %arg3: !spirv.arm.tensor<12x13x3x14xi8>) -> (!spirv.arm.tensor<12x13x12x14xi8>) {
    %0 = spirv.Constant 2 : i32
    // CHECK: {{%.*}} = spirv.Tosa.Concat %arg0, %arg1, %arg2, %arg3 {axis = 2 : i32} : !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8> -> !spirv.arm.tensor<12x13x12x14xi8>
    %1 = spirv.Tosa.Concat %arg0, %arg1, %arg2, %arg3 {axis = 2 : i32} : !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8>, !spirv.arm.tensor<12x13x3x14xi8> -> !spirv.arm.tensor<12x13x12x14xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<12x13x12x14xi8>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<12x13x12x14xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Concat - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @concat_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<40x31x19xf32>, UniformConstant>
  spirv.GlobalVariable @concat_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<40x15x19xf32>, UniformConstant>
  spirv.GlobalVariable @concat_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<40x16x19xf32>, UniformConstant>
  spirv.GlobalVariable @concat_fp_res_0 bind(0, 3) : !spirv.ptr<!spirv.arm.tensor<40x62x19xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @concat_fp, @concat_fp_arg_0, @concat_fp_arg_1, @concat_fp_arg_2, @concat_fp_res_0
  spirv.ARM.Graph @concat_fp(%arg0: !spirv.arm.tensor<40x31x19xf32>, %arg1: !spirv.arm.tensor<40x15x19xf32>, %arg2: !spirv.arm.tensor<40x16x19xf32>) -> (!spirv.arm.tensor<40x62x19xf32>) {
    %0 = spirv.Constant 1 : i32
    // CHECK: {{%.*}} = spirv.Tosa.Concat %arg0, %arg1, %arg2 {axis = 1 : i32} : !spirv.arm.tensor<40x31x19xf32>, !spirv.arm.tensor<40x15x19xf32>, !spirv.arm.tensor<40x16x19xf32> -> !spirv.arm.tensor<40x62x19xf32>
    %1 = spirv.Tosa.Concat %arg0, %arg1, %arg2 {axis = 1 : i32} : !spirv.arm.tensor<40x31x19xf32>, !spirv.arm.tensor<40x15x19xf32>, !spirv.arm.tensor<40x16x19xf32> -> !spirv.arm.tensor<40x62x19xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<40x62x19xf32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<40x62x19xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pad - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @pad_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<4x7xi8>, UniformConstant>
  spirv.GlobalVariable @pad_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<21x19xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @pad_int, @pad_int_arg_0, @pad_int_res_0
  spirv.ARM.Graph @pad_int(%arg0: !spirv.arm.tensor<4x7xi8>) -> (!spirv.arm.tensor<21x19xi8>) {
    %0 = spirv.Constant dense<[10, 7, 6, 6]> : !spirv.arm.tensor<4xi32>
    %1 = spirv.Constant dense<-76> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Pad %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<4x7xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<21x19xi8>
    %2 = spirv.Tosa.Pad %arg0, %0, %1 : !spirv.arm.tensor<4x7xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<21x19xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<21x19xi8>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<21x19xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pad - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @pad_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x9x2x3xf32>, UniformConstant>
  spirv.GlobalVariable @pad_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<4x9x4x4xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @pad_fp, @pad_fp_arg_0, @pad_fp_res_0
  spirv.ARM.Graph @pad_fp(%arg0: !spirv.arm.tensor<2x9x2x3xf32>) -> (!spirv.arm.tensor<4x9x4x4xf32>) {
    %0 = spirv.Constant dense<[1, 1, 0, 0, 1, 1, 0, 1]> : !spirv.arm.tensor<8xi32>
    %1 = spirv.Constant dense<1.21630913E+38> : !spirv.arm.tensor<1xf32>
    // CHECK: {{%.*}} = spirv.Tosa.Pad %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<2x9x2x3xf32>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<4x9x4x4xf32>
    %2 = spirv.Tosa.Pad %arg0, %0, %1 : !spirv.arm.tensor<2x9x2x3xf32>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<4x9x4x4xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x9x4x4xf32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<4x9x4x4xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reshape - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reshape_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<25x6x29x35xi16>, UniformConstant>
  spirv.GlobalVariable @reshape_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<125x6x7x29xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reshape_int, @reshape_int_arg_0, @reshape_int_res_0
  spirv.ARM.Graph @reshape_int(%arg0: !spirv.arm.tensor<25x6x29x35xi16>) -> (!spirv.arm.tensor<125x6x7x29xi16>) {
    %0 = spirv.Constant dense<[125, 6, 7, 29]> : !spirv.arm.tensor<4xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Reshape %arg0, {{%.*}} : !spirv.arm.tensor<25x6x29x35xi16>, !spirv.arm.tensor<4xi32> -> !spirv.arm.tensor<125x6x7x29xi16>
    %1 = spirv.Tosa.Reshape %arg0, %0 : !spirv.arm.tensor<25x6x29x35xi16>, !spirv.arm.tensor<4xi32> -> !spirv.arm.tensor<125x6x7x29xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<125x6x7x29xi16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<125x6x7x29xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reshape - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reshape_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x2x7x2xf32>, UniformConstant>
  spirv.GlobalVariable @reshape_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<2x1x14xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reshape_fp, @reshape_fp_arg_0, @reshape_fp_res_0
  spirv.ARM.Graph @reshape_fp(%arg0: !spirv.arm.tensor<1x2x7x2xf32>) -> (!spirv.arm.tensor<2x1x14xf32>) {
    %0 = spirv.Constant dense<[2, 1, 14]> : !spirv.arm.tensor<3xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Reshape %arg0, {{%.*}} : !spirv.arm.tensor<1x2x7x2xf32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x1x14xf32>
    %1 = spirv.Tosa.Reshape %arg0, %0 : !spirv.arm.tensor<1x2x7x2xf32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<2x1x14xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x1x14xf32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<2x1x14xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reverse - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reverse_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<20x5x28x31xi32>, UniformConstant>
  spirv.GlobalVariable @reverse_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<20x5x28x31xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reverse_int, @reverse_int_arg_0, @reverse_int_res_0
  spirv.ARM.Graph @reverse_int(%arg0: !spirv.arm.tensor<20x5x28x31xi32>) -> (!spirv.arm.tensor<20x5x28x31xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Reverse %arg0 {axis = 2 : i32} : !spirv.arm.tensor<20x5x28x31xi32> -> !spirv.arm.tensor<20x5x28x31xi32>
    %1 = spirv.Tosa.Reverse %arg0 {axis = 2 : i32} : !spirv.arm.tensor<20x5x28x31xi32> -> !spirv.arm.tensor<20x5x28x31xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<20x5x28x31xi32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<20x5x28x31xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reverse - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @reverse_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<21x34x47xf32>, UniformConstant>
  spirv.GlobalVariable @reverse_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<21x34x47xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @reverse_fp, @reverse_fp_arg_0, @reverse_fp_res_0
  spirv.ARM.Graph @reverse_fp(%arg0: !spirv.arm.tensor<21x34x47xf32>) -> (!spirv.arm.tensor<21x34x47xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Reverse %arg0 {axis = 1 : i32} : !spirv.arm.tensor<21x34x47xf32> -> !spirv.arm.tensor<21x34x47xf32>
    %1 = spirv.Tosa.Reverse %arg0 {axis = 1 : i32} : !spirv.arm.tensor<21x34x47xf32> -> !spirv.arm.tensor<21x34x47xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<21x34x47xf32>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<21x34x47xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Slice - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @slice_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<32x19x41xi8>, UniformConstant>
  spirv.GlobalVariable @slice_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<21x5x2xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @slice_int, @slice_int_arg_0, @slice_int_res_0
  spirv.ARM.Graph @slice_int(%arg0: !spirv.arm.tensor<32x19x41xi8>) -> (!spirv.arm.tensor<21x5x2xi8>) {
    %0 = spirv.Constant dense<[8, 11, 39]> : !spirv.arm.tensor<3xi32>
    %1 = spirv.Constant dense<[21, 5, 2]> : !spirv.arm.tensor<3xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Slice %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<32x19x41xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<21x5x2xi8>
    %2 = spirv.Tosa.Slice %arg0, %0, %1 : !spirv.arm.tensor<32x19x41xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<21x5x2xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<21x5x2xi8>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<21x5x2xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Slice - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @slice_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<30x45x29xf32>, UniformConstant>
  spirv.GlobalVariable @slice_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<5x12x11xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @slice_fp, @slice_fp_arg_0, @slice_fp_res_0
  spirv.ARM.Graph @slice_fp(%arg0: !spirv.arm.tensor<30x45x29xf32>) -> (!spirv.arm.tensor<5x12x11xf32>) {
    %0 = spirv.Constant dense<[21, 20, 10]> : !spirv.arm.tensor<3xi32>
    %1 = spirv.Constant dense<[5, 12, 11]> : !spirv.arm.tensor<3xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Slice %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<30x45x29xf32>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<5x12x11xf32>
    %2 = spirv.Tosa.Slice %arg0, %0, %1 : !spirv.arm.tensor<30x45x29xf32>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<5x12x11xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<5x12x11xf32>
    spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<5x12x11xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tile - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @tile_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<10x28x21xi16>, UniformConstant>
  spirv.GlobalVariable @tile_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<10x28x63xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @tile_int, @tile_int_arg_0, @tile_int_res_0
  spirv.ARM.Graph @tile_int(%arg0: !spirv.arm.tensor<10x28x21xi16>) -> (!spirv.arm.tensor<10x28x63xi16>) {
    %0 = spirv.Constant dense<[1, 1, 3]> : !spirv.arm.tensor<3xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Tile %arg0, {{%.*}} : !spirv.arm.tensor<10x28x21xi16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<10x28x63xi16>
    %1 = spirv.Tosa.Tile %arg0, %0 : !spirv.arm.tensor<10x28x21xi16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<10x28x63xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<10x28x63xi16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<10x28x63xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tile - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @tile_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<31x19x5xf16>, UniformConstant>
  spirv.GlobalVariable @tile_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<62x57x10xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @tile_fp, @tile_fp_arg_0, @tile_fp_res_0
  spirv.ARM.Graph @tile_fp(%arg0: !spirv.arm.tensor<31x19x5xf16>) -> (!spirv.arm.tensor<62x57x10xf16>) {
    %0 = spirv.Constant dense<[2, 3, 2]> : !spirv.arm.tensor<3xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Tile %arg0, {{%.*}} : !spirv.arm.tensor<31x19x5xf16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<62x57x10xf16>
    %1 = spirv.Tosa.Tile %arg0, %0 : !spirv.arm.tensor<31x19x5xf16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<62x57x10xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<62x57x10xf16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<62x57x10xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Transpose - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @transpose_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x28x1x61xi16>, UniformConstant>
  spirv.GlobalVariable @transpose_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x14x28x61xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @transpose_int, @transpose_int_arg_0, @transpose_int_res_0
  spirv.ARM.Graph @transpose_int(%arg0: !spirv.arm.tensor<14x28x1x61xi16>) -> (!spirv.arm.tensor<1x14x28x61xi16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Transpose %arg0 {perms = dense<[2, 0, 1, 3]> : !spirv.arm.tensor<4xi32>} : !spirv.arm.tensor<14x28x1x61xi16> -> !spirv.arm.tensor<1x14x28x61xi16>
    %1 = spirv.Tosa.Transpose %arg0 {perms = dense<[2, 0, 1, 3]> : !spirv.arm.tensor<4xi32>} : !spirv.arm.tensor<14x28x1x61xi16> -> !spirv.arm.tensor<1x14x28x61xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x14x28x61xi16>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<1x14x28x61xi16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Transpose - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @transpose_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<42x22x49xi1>, UniformConstant>
  spirv.GlobalVariable @transpose_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<49x42x22xi1>, UniformConstant>
  spirv.ARM.GraphEntryPoint @transpose_fp, @transpose_fp_arg_0, @transpose_fp_res_0
  spirv.ARM.Graph @transpose_fp(%arg0: !spirv.arm.tensor<42x22x49xi1>) -> (!spirv.arm.tensor<49x42x22xi1>) {
    // CHECK: {{%.*}} = spirv.Tosa.Transpose %arg0 {perms = dense<[2, 0, 1]> : !spirv.arm.tensor<3xi32>} : !spirv.arm.tensor<42x22x49xi1> -> !spirv.arm.tensor<49x42x22xi1>
    %1 = spirv.Tosa.Transpose %arg0 {perms = dense<[2, 0, 1]> : !spirv.arm.tensor<3xi32>} : !spirv.arm.tensor<42x22x49xi1> -> !spirv.arm.tensor<49x42x22xi1>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<49x42x22xi1>
    spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<49x42x22xi1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Gather - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @gather_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<31x11x45xi32>, UniformConstant>
  spirv.GlobalVariable @gather_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<31x15xi32>, UniformConstant>
  spirv.GlobalVariable @gather_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<31x15x45xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @gather_int, @gather_int_arg_0, @gather_int_arg_1, @gather_int_res_0
  spirv.ARM.Graph @gather_int(%arg0: !spirv.arm.tensor<31x11x45xi32>, %arg1: !spirv.arm.tensor<31x15xi32>) -> (!spirv.arm.tensor<31x15x45xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<31x15x45xi32>
    %0 = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<31x15x45xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<31x15x45xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<31x15x45xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Gather - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @gather_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<59x61x19xf32>, UniformConstant>
  spirv.GlobalVariable @gather_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<59x65xi32>, UniformConstant>
  spirv.GlobalVariable @gather_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<59x65x19xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @gather_fp, @gather_fp_arg_0, @gather_fp_arg_1, @gather_fp_res_0
  spirv.ARM.Graph @gather_fp(%arg0: !spirv.arm.tensor<59x61x19xf32>, %arg1: !spirv.arm.tensor<59x65xi32>) -> (!spirv.arm.tensor<59x65x19xf32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<59x61x19xf32>, !spirv.arm.tensor<59x65xi32> -> !spirv.arm.tensor<59x65x19xf32>
    %0 = spirv.Tosa.Gather %arg0, %arg1 : !spirv.arm.tensor<59x61x19xf32>, !spirv.arm.tensor<59x65xi32> -> !spirv.arm.tensor<59x65x19xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<59x65x19xf32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<59x65x19xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Scatter - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @scatter_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<34x28x54xi32>, UniformConstant>
  spirv.GlobalVariable @scatter_int_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<34x18xi32>, UniformConstant>
  spirv.GlobalVariable @scatter_int_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<34x18x54xi32>, UniformConstant>
  spirv.GlobalVariable @scatter_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<34x28x54xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @scatter_int, @scatter_int_arg_0, @scatter_int_arg_1, @scatter_int_arg_2, @scatter_int_res_0
  spirv.ARM.Graph @scatter_int(%arg0: !spirv.arm.tensor<34x28x54xi32>, %arg1: !spirv.arm.tensor<34x18xi32>, %arg2: !spirv.arm.tensor<34x18x54xi32>) -> (!spirv.arm.tensor<34x28x54xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi32> -> !spirv.arm.tensor<34x28x54xi32>
    %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi32> -> !spirv.arm.tensor<34x28x54xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<34x28x54xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<34x28x54xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Scatter - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @scatter_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<18x34x25xf16>, UniformConstant>
  spirv.GlobalVariable @scatter_fp_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<18x20xi32>, UniformConstant>
  spirv.GlobalVariable @scatter_fp_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<18x20x25xf16>, UniformConstant>
  spirv.GlobalVariable @scatter_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<18x34x25xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @scatter_fp, @scatter_fp_arg_0, @scatter_fp_arg_1, @scatter_fp_arg_2, @scatter_fp_res_0
  spirv.ARM.Graph @scatter_fp(%arg0: !spirv.arm.tensor<18x34x25xf16>, %arg1: !spirv.arm.tensor<18x20xi32>, %arg2: !spirv.arm.tensor<18x20x25xf16>) -> (!spirv.arm.tensor<18x34x25xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<18x34x25xf16>, !spirv.arm.tensor<18x20xi32>, !spirv.arm.tensor<18x20x25xf16> -> !spirv.arm.tensor<18x34x25xf16>
    %0 = spirv.Tosa.Scatter %arg0, %arg1, %arg2 : !spirv.arm.tensor<18x34x25xf16>, !spirv.arm.tensor<18x20xi32>, !spirv.arm.tensor<18x20x25xf16> -> !spirv.arm.tensor<18x34x25xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<18x34x25xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<18x34x25xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Resize - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @resize_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x1x31x55xi8>, UniformConstant>
  spirv.GlobalVariable @resize_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x1x278x55xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @resize_int, @resize_int_arg_0, @resize_int_res_0
  spirv.ARM.Graph @resize_int(%arg0: !spirv.arm.tensor<1x1x31x55xi8>) -> (!spirv.arm.tensor<1x1x278x55xi8>) {
    %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
    %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
    %3 = spirv.Constant dense<[0, 7]> : !spirv.arm.tensor<2xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Resize %arg0, {{%.*}}, {{%.*}}, {{%.*}} {mode = #spirv.tosa_ext_resize_mode_type<NearestNeighbor>} : !spirv.arm.tensor<1x1x31x55xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xi8>
    %4 = spirv.Tosa.Resize %arg0, %1, %2, %3 {mode = #spirv.tosa_ext_resize_mode_type<NearestNeighbor>} : !spirv.arm.tensor<1x1x31x55xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xi8>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x1x278x55xi8>
    spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x1x278x55xi8>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Resize - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @resize_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x48x33x63xf32>, UniformConstant>
  spirv.GlobalVariable @resize_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x753x297x63xf32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @resize_fp, @resize_fp_arg_0, @resize_fp_res_0
  spirv.ARM.Graph @resize_fp(%arg0: !spirv.arm.tensor<1x48x33x63xf32>) -> (!spirv.arm.tensor<1x753x297x63xf32>) {
    %1 = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
    %2 = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
    %3 = spirv.Constant dense<[0, 8]> : !spirv.arm.tensor<2xi32>
    // CHECK: {{%.*}} = spirv.Tosa.Resize %arg0, {{%.*}}, {{%.*}}, {{%.*}} {mode = #spirv.tosa_ext_resize_mode_type<Bilinear>} : !spirv.arm.tensor<1x48x33x63xf32>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf32>
    %4 = spirv.Tosa.Resize %arg0, %1, %2, %3 {mode = #spirv.tosa_ext_resize_mode_type<Bilinear>} : !spirv.arm.tensor<1x48x33x63xf32>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x753x297x63xf32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x753x297x63xf32>
    spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x753x297x63xf32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cast - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @cast_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x65538x1x2xi8>, UniformConstant>
  spirv.GlobalVariable @cast_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<1x65538x1x2xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @cast_int, @cast_int_arg_0, @cast_int_res_0
  spirv.ARM.Graph @cast_int(%arg0: !spirv.arm.tensor<1x65538x1x2xi8>) -> (!spirv.arm.tensor<1x65538x1x2xi32>) {
    // CHECK: {{%.*}} = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<1x65538x1x2xi8> -> !spirv.arm.tensor<1x65538x1x2xi32>
    %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<1x65538x1x2xi8> -> !spirv.arm.tensor<1x65538x1x2xi32>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65538x1x2xi32>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1x65538x1x2xi32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cast - PRO-FP
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @cast_fp_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<11x5x14x4xf32>, UniformConstant>
  spirv.GlobalVariable @cast_fp_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<11x5x14x4xf16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @cast_fp, @cast_fp_arg_0, @cast_fp_res_0
  spirv.ARM.Graph @cast_fp(%arg0: !spirv.arm.tensor<11x5x14x4xf32>) -> (!spirv.arm.tensor<11x5x14x4xf16>) {
    // CHECK: {{%.*}} = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<11x5x14x4xf32> -> !spirv.arm.tensor<11x5x14x4xf16>
    %0 = spirv.Tosa.Cast %arg0 : !spirv.arm.tensor<11x5x14x4xf32> -> !spirv.arm.tensor<11x5x14x4xf16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x5x14x4xf16>
    spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x5x14x4xf16>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rescale - PRO-INT
//===----------------------------------------------------------------------===//

// CHECK: spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>
spirv.module Logical Vulkan requires #spirv.vce<v1.3, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @rescale_int_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<17x29x19xi16>, UniformConstant>
  spirv.GlobalVariable @rescale_int_res_0 bind(1, 0) : !spirv.ptr<!spirv.arm.tensor<17x29x19xi16>, UniformConstant>
  spirv.ARM.GraphEntryPoint @rescale_int, @rescale_int_arg_0, @rescale_int_res_0
  spirv.ARM.Graph @rescale_int(%arg0: !spirv.arm.tensor<17x29x19xi16>) -> (!spirv.arm.tensor<17x29x19xi16>) {
    %5 = spirv.Constant dense<1866149760> : !spirv.arm.tensor<1xi32>
    %6 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
    %7 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
    %8 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
    // CHECK: {{%.*}} = spirv.Tosa.Rescale %arg0, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = #spirv.tosa_ext_rounding_mode_type<DoubleRound>, scale32 = true} : !spirv.arm.tensor<17x29x19xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<17x29x19xi16>
    %9 = spirv.Tosa.Rescale %arg0, %5, %6, %7, %8 {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = #spirv.tosa_ext_rounding_mode_type<DoubleRound>, scale32 = true} : !spirv.arm.tensor<17x29x19xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<17x29x19xi16>
    // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<17x29x19xi16>
    spirv.ARM.GraphOutputs %9 : !spirv.arm.tensor<17x29x19xi16>
  }
}
