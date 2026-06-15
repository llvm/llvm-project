// RUN: mlir-opt --split-input-file --spirv-lower-abi-attrs --verify-diagnostics %s \
// RUN:   | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.module
spirv.module Logical GLSL450 {
  //  CHECK-DAG:    spirv.GlobalVariable [[VAR0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer>
  //  CHECK-DAG:    spirv.GlobalVariable [[VAR1:@.*]] bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
  //      CHECK:    spirv.func [[FN:@.*]]()
  // We cannot generate SubgroupSize execution mode for Shader capability -- leave it alone.
  // CHECK-SAME:      #spirv.entry_point_abi<subgroup_size = 64>
  spirv.func @kernel(
    %arg0: f32
           {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>},
    %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, StorageBuffer>
           {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) "None"
  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1], subgroup_size = 64>} {
    // CHECK: [[ADDRESSARG0:%.*]] = spirv.mlir.addressof [[VAR0]]
    // CHECK: [[CONST0:%.*]] = spirv.Constant 0 : i32
    // CHECK: [[ARG0PTR:%.*]] = spirv.AccessChain [[ADDRESSARG0]]{{\[}}[[CONST0]]
    // CHECK: [[ARG0:%.*]] = spirv.Load "StorageBuffer" [[ARG0PTR]]
    // CHECK: [[ARG1:%.*]] = spirv.mlir.addressof [[VAR1]]
    // CHECK: spirv.Return
    spirv.Return
  }
  // CHECK: spirv.EntryPoint "GLCompute" [[FN]]
  // CHECK: spirv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
} // end spirv.module

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
     #spirv.vce<v1.0, [VulkanMemoryModel, Shader, Int8, TensorsARM, GraphARM], [SPV_ARM_tensors, SPV_ARM_graph, SPV_KHR_vulkan_memory_model]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.module
spirv.module Logical Vulkan {
  //  CHECK-DAG:    spirv.GlobalVariable [[VARARG0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<1x16x16x16xi8>, UniformConstant>
  //  CHECK-DAG:    spirv.GlobalVariable [[VARRES0:@.*]] bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<1x16x16x16xi8>, UniformConstant>

  //      CHECK:    spirv.ARM.GraphEntryPoint [[GN:@.*]], [[VARARG0]], [[VARRES0]]
  //      CHECK:    spirv.ARM.Graph [[GN]]([[ARG0:%.*]]: !spirv.arm.tensor<1x16x16x16xi8>) -> !spirv.arm.tensor<1x16x16x16xi8> attributes {entry_point = true}
  spirv.ARM.Graph @main(%arg0: !spirv.arm.tensor<1x16x16x16xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>})
                  -> (!spirv.arm.tensor<1x16x16x16xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
    spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<1x16x16x16xi8>
  }
} // end spirv.module

} // end module

// -----

module {
// expected-error@+1 {{'spirv.module' op missing SPIR-V target env attribute}}
spirv.module Logical GLSL450 {}
} // end module

// -----

// CHECK-LABEL: spirv.module
// Test case with SPIRV version 1.4: all the interface's storage variables are passed to OpEntryPoint
spirv.module Logical GLSL450 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
  //  CHECK-DAG:    spirv.GlobalVariable [[VAR0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer>
  //  CHECK-DAG:    spirv.GlobalVariable [[VAR1:@.*]] bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
  //      CHECK:    spirv.func [[FN:@.*]]()
  // CHECK-SAME:      #spirv.entry_point_abi<subgroup_size = 64>
  spirv.func @kernel(
    %arg0: f32
           {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>},
    %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, StorageBuffer>
           {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) "None"
  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1], subgroup_size = 64>} {
    // CHECK: [[ADDRESSARG0:%.*]] = spirv.mlir.addressof [[VAR0]]
    // CHECK: [[CONST0:%.*]] = spirv.Constant 0 : i32
    // CHECK: [[ARG0PTR:%.*]] = spirv.AccessChain [[ADDRESSARG0]]{{\[}}[[CONST0]]
    // CHECK: [[ARG0:%.*]] = spirv.Load "StorageBuffer" [[ARG0PTR]]
    // CHECK: [[ARG1:%.*]] = spirv.mlir.addressof [[VAR1]]
    // CHECK: spirv.Return
    spirv.Return
  }
  // CHECK: spirv.EntryPoint "GLCompute" [[FN]], [[VAR0]], [[VAR1]]
  // CHECK: spirv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
} // end spirv.module

// -----

module {
  spirv.module Logical GLSL450 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Sampled1D], []>, #spirv.resource_limits<>>} {
    // CHECK-DAG: spirv.GlobalVariable @[[IMAGE_GV:.*]] bind(0, 0) : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>, UniformConstant>
    // CHECK: spirv.func @read_image
    spirv.func @read_image(%arg0: !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>, UniformConstant> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}, %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) "None" attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      // CHECK: %[[IMAGE_ADDR:.*]] = spirv.mlir.addressof @[[IMAGE_GV]] : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>, UniformConstant>
      %cst0_i32 = spirv.Constant 0 : i32
      // CHECK: spirv.Load "UniformConstant" %[[IMAGE_ADDR]]
      %0 = spirv.Load "UniformConstant" %arg0 : !spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
      %1 = spirv.Image %0 : !spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
      %2 = spirv.ImageFetch %1, %cst0_i32  : !spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>, i32 -> vector<4xf32>
      %3 = spirv.CompositeExtract %2[0 : i32] : vector<4xf32>
      %cst0_i32_0 = spirv.Constant 0 : i32
      %cst0_i32_1 = spirv.Constant 0 : i32
      %cst1_i32 = spirv.Constant 1 : i32
      %4 = spirv.AccessChain %arg1[%cst0_i32_0, %cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      spirv.Store "StorageBuffer" %4, %3 : f32
      spirv.Return
    }
  }
}
