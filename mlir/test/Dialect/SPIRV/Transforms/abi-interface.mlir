// RUN: mlir-opt -spirv-lower-abi-attrs -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.module
spirv.module Logical GLSL450 {
  // CHECK-DAG:    spirv.GlobalVariable [[VAR0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer>
  // CHECK-DAG:    spirv.GlobalVariable [[VAR1:@.*]] bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:    spirv.func [[FN:@.*]]()
  spirv.func @kernel(
    %arg0: f32
           {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>},
    %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, StorageBuffer>
           {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) "None"
  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 1, 1]> : vector<3xi32>>} {
    // CHECK: [[ARG1:%.*]] = spirv.mlir.addressof [[VAR1]]
    // CHECK: [[ADDRESSARG0:%.*]] = spirv.mlir.addressof [[VAR0]]
    // CHECK: [[CONST0:%.*]] = spirv.Constant 0 : i32
    // CHECK: [[ARG0PTR:%.*]] = spirv.AccessChain [[ADDRESSARG0]]{{\[}}[[CONST0]]
    // CHECK: [[ARG0:%.*]] = spirv.Load "StorageBuffer" [[ARG0PTR]]
    // CHECK: spirv.Return
    spirv.Return
  }
  // CHECK: spirv.EntryPoint "GLCompute" [[FN]]
  // CHECK: spirv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
} // end spirv.module

} // end module
