// RUN: mlir-translate -no-implicit-module -split-input-file --verify-diagnostics -mlir-print-debuginfo -test-spirv-roundtrip-debug %s | FileCheck %s
// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --spirv-emit-debug-info --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

//===----------------------------------------------------------------------===//
// spirv DebugInfo: FileLineCol Locations
//===----------------------------------------------------------------------===//

// CHECK: #loc[[LOC_TENSOR0:.*]] = loc("{{.*}}debug-info.mlir{{.*}}:21:27")
// CHECK: #loc[[LOC_TENSOR1:.*]] = loc("{{.*}}debug-info.mlir{{.*}}:21:67")
// CHECK: #loc[[LOC_TENSOR2:.*]] = loc("{{.*}}debug-info.mlir{{.*}}:21:105")
spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_non_semantic_info]> {
  spirv.GlobalVariable @conv2d_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x16x16x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv2d_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<8x3x3x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv2d_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<8xi32>, UniformConstant>
  spirv.GlobalVariable @conv2d_res_0 bind(0, 3) : !spirv.ptr<!spirv.arm.tensor<1x14x14x8xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @conv2d, @conv2d_arg_0, @conv2d_arg_1, @conv2d_arg_2, @conv2d_res_0
  // CHECK: spirv.ARM.Graph @{{.*}}(%arg0: !spirv.arm.tensor<1x16x16x1xi8> loc("{{.*}}debug-info.mlir{{.*}}:21:27"), %arg1: !spirv.arm.tensor<8x3x3x1xi8> loc("{{.*}}debug-info.mlir{{.*}}:21:67"), %arg2: !spirv.arm.tensor<8xi32> loc("{{.*}}debug-info.mlir{{.*}}:21:105")) -> !spirv.arm.tensor<1x14x14x8xi32> attributes {entry_point = true} {
  spirv.ARM.Graph @conv2d(%arg0: !spirv.arm.tensor<1x16x16x1xi8>, %arg1: !spirv.arm.tensor<8x3x3x1xi8>, %arg2: !spirv.arm.tensor<8xi32>) -> (!spirv.arm.tensor<1x14x14x8xi32>) {
      %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
      %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
      // CHECK: {{%.*}} = spirv.Tosa.Conv2D{{.*}}loc(#loc[[LOC_OP:.*]])
      %2 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %0, %1 : !spirv.arm.tensor<1x16x16x1xi8>, !spirv.arm.tensor<8x3x3x1xi8>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x14x14x8xi32>
      spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<1x14x14x8xi32>
  // CHECK: } loc(#loc[[LOC_GRAPH:.*]])
  }
}
// CHECK-DAG: #loc[[LOC_GRAPH]] = loc("{{.*}}debug-info.mlir{{.*}}:21:3")
// CHECK-DAG: #loc[[LOC_OP]] = loc("{{.*}}debug-info.mlir{{.*}}:25:12")

// -----

//===----------------------------------------------------------------------===//
// spirv DebugInfo: Name Locations
//===----------------------------------------------------------------------===//

// CHECK: #loc[[NAME_TENSOR0:.*]] = loc("tensor_0")
// CHECK: #loc[[NAME_TENSOR1:.*]] = loc("tensor_1")
// CHECK: #loc[[NAME_TENSOR2:.*]] = loc("tensor_2")
spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_non_semantic_info]> {
  spirv.GlobalVariable @conv2d_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x16x16x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv2d_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<8x3x3x1xi8>, UniformConstant>
  spirv.GlobalVariable @conv2d_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<8xi32>, UniformConstant>
  spirv.GlobalVariable @conv2d_res_0 bind(0, 3) : !spirv.ptr<!spirv.arm.tensor<1x14x14x8xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @conv2d, @conv2d_arg_0, @conv2d_arg_1, @conv2d_arg_2, @conv2d_res_0
  // CHECK: spirv.ARM.Graph @{{.*}}(%arg0: !spirv.arm.tensor<1x16x16x1xi8> loc("tensor_0"), %arg1: !spirv.arm.tensor<8x3x3x1xi8> loc("tensor_1"), %arg2: !spirv.arm.tensor<8xi32> loc("tensor_2")) -> !spirv.arm.tensor<1x14x14x8xi32> attributes {entry_point = true} {
  spirv.ARM.Graph @conv2d(%arg0: !spirv.arm.tensor<1x16x16x1xi8> loc("tensor_0") , %arg1: !spirv.arm.tensor<8x3x3x1xi8> loc("tensor_1"), %arg2: !spirv.arm.tensor<8xi32> loc("tensor_2")) -> (!spirv.arm.tensor<1x14x14x8xi32>) {
      %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
      %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
      // CHECK: {{%.*}} = spirv.Tosa.Conv2D{{.*}}loc(#loc[[NAME_OP:.*]])
      %2 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %0, %1 : !spirv.arm.tensor<1x16x16x1xi8>, !spirv.arm.tensor<8x3x3x1xi8>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x14x14x8xi32> loc("op_0")
      spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<1x14x14x8xi32>
  // CHECK: } loc(#loc[[NAME_GRAPH:.*]])
  } loc("graph_0")
}
// CHECK-DAG: #loc[[NAME_GRAPH]] = loc("graph_0")
// CHECK-DAG: #loc[[NAME_OP]] = loc("op_0")

// -----

//===----------------------------------------------------------------------===//
// spirv DebugInfo: Multiple tosa ops with same locations
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_non_semantic_info]> {
  spirv.GlobalVariable @test_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x9x3x32xi16>, UniformConstant>
  spirv.GlobalVariable @test_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x9x3x32xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @test, @test_arg_0, @test_res_0
  spirv.ARM.Graph @test(%arg0: !spirv.arm.tensor<2x9x3x32xi16> loc("tensor_0")) -> (!spirv.arm.tensor<2x9x3x32xi8>) attributes {entry_point = true} {
    %weight = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<32x1x1x32xi8>
    %bias = spirv.ARM.GraphConstant {graph_constant_id = 1 : i32} : !spirv.arm.tensor<32xi64>
    %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
    %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
    // CHECK: %[[CONV2D:.*]] = spirv.Tosa.Conv2D{{.*}}loc(#loc[[SAME_OP:.*]])
    %conv2d = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %weight, %bias, %0, %1  : !spirv.arm.tensor<2x9x3x32xi16>, !spirv.arm.tensor<32x1x1x32xi8>, !spirv.arm.tensor<32xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x9x3x32xi64> loc("op_0")
    %multiplier = spirv.ARM.GraphConstant {graph_constant_id = 2 : i32} : !spirv.arm.tensor<1xi16>
    %shift = spirv.ARM.GraphConstant {graph_constant_id = 3 : i32} : !spirv.arm.tensor<1xi8>
    %5 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
    %6 = spirv.Constant dense<-4> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Rescale{{.*}}loc(#loc[[SAME_OP]])
    %rescale = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %conv2d, %multiplier, %shift, %5, %6 : !spirv.arm.tensor<2x9x3x32xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x9x3x32xi8> loc("op_0")
    spirv.ARM.GraphOutputs %rescale : !spirv.arm.tensor<2x9x3x32xi8>
  // CHECK: } loc(#loc[[SAME_GRAPH:.*]])
  } loc("graph_0")
}
// CHECK-DAG: #loc[[SAME_GRAPH]] = loc("graph_0")
// CHECK-DAG: #loc[[SAME_OP]] = loc("op_0")

// -----

//===----------------------------------------------------------------------===//
// spirv DebugInfo: Multiple tosa ops with differing locations
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_non_semantic_info]> {
  spirv.GlobalVariable @test_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<2x9x3x32xi16>, UniformConstant>
  spirv.GlobalVariable @test_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<2x9x3x32xi8>, UniformConstant>
  spirv.ARM.GraphEntryPoint @test, @test_arg_0, @test_res_0
  spirv.ARM.Graph @test(%arg0: !spirv.arm.tensor<2x9x3x32xi16> loc("tensor_0")) -> (!spirv.arm.tensor<2x9x3x32xi8>) attributes {entry_point = true} {
    %weight = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<32x1x1x32xi8>
    %bias = spirv.ARM.GraphConstant {graph_constant_id = 1 : i32} : !spirv.arm.tensor<32xi64>
    %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
    %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
    // CHECK: %[[CONV2D:.*]] = spirv.Tosa.Conv2D{{.*}}loc(#loc[[MULTI_OP0:.*]])
    %conv2d = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %weight, %bias, %0, %1  : !spirv.arm.tensor<2x9x3x32xi16>, !spirv.arm.tensor<32x1x1x32xi8>, !spirv.arm.tensor<32xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x9x3x32xi64> loc("op_0")
    %multiplier = spirv.ARM.GraphConstant {graph_constant_id = 2 : i32} : !spirv.arm.tensor<1xi16>
    %shift = spirv.ARM.GraphConstant {graph_constant_id = 3 : i32} : !spirv.arm.tensor<1xi8>
    %5 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi64>
    %6 = spirv.Constant dense<-4> : !spirv.arm.tensor<1xi8>
    // CHECK: {{%.*}} = spirv.Tosa.Rescale{{.*}}loc(#loc[[MULTI_OP1:.*]])
    %rescale = spirv.Tosa.Rescale scale32 = false, rounding_mode = <SingleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %conv2d, %multiplier, %shift, %5, %6 : !spirv.arm.tensor<2x9x3x32xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x9x3x32xi8> loc("op_1")
    spirv.ARM.GraphOutputs %rescale : !spirv.arm.tensor<2x9x3x32xi8>
  // CHECK: } loc(#loc[[MULTI_GRAPH:.*]])
  } loc("graph_0")
}
// CHECK-DAG: #loc[[MULTI_GRAPH]] = loc("graph_0")
// CHECK-DAG: #loc[[MULTI_OP0]] = loc("op_0")
// CHECK-DAG: #loc[[MULTI_OP1]] = loc("op_1")
// -----

//===----------------------------------------------------------------------===//
// spirv DebugInfo: Fused Locations
//===----------------------------------------------------------------------===//

spirv.module Logical Vulkan requires #spirv.vce<v1.6, [VulkanMemoryModel, Shader, Int8, Int16, Int64, Float16, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_non_semantic_info]> {
  spirv.GlobalVariable @fused_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x16x16x1xi8>, UniformConstant>
  spirv.GlobalVariable @fused_arg_1 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<8x3x3x1xi8>, UniformConstant>
  spirv.GlobalVariable @fused_arg_2 bind(0, 2) : !spirv.ptr<!spirv.arm.tensor<8xi32>, UniformConstant>
  spirv.GlobalVariable @fused_res_0 bind(0, 3) : !spirv.ptr<!spirv.arm.tensor<1x14x14x8xi32>, UniformConstant>
  spirv.ARM.GraphEntryPoint @fused, @fused_arg_0, @fused_arg_1, @fused_arg_2, @fused_res_0
  spirv.ARM.Graph @fused(%arg0: !spirv.arm.tensor<1x16x16x1xi8> loc("tensor_0"), %arg1: !spirv.arm.tensor<8x3x3x1xi8>, %arg2: !spirv.arm.tensor<8xi32>) -> (!spirv.arm.tensor<1x14x14x8xi32>) {
      %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
      %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
      // CHECK: {{%.*}} = spirv.Tosa.Conv2D{{.*}}loc(#loc[[FUSED:.*]])
      %2 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %0, %1 : !spirv.arm.tensor<1x16x16x1xi8>, !spirv.arm.tensor<8x3x3x1xi8>, !spirv.arm.tensor<8xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x14x14x8xi32> loc(fused["op_0", "source.cc":12:34])
      spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<1x14x14x8xi32>
  // CHECK: } loc(#loc[[FUSED_GRAPH:.*]])
  } loc("graph_0")
}
// CHECK-DAG: #loc[[FUSED_GRAPH]] = loc("graph_0")
// CHECK-DAG: #loc[[FUSED]] = loc("op_0;source.cc:12:34")
