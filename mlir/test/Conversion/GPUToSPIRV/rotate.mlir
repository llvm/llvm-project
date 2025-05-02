// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformRotateKHR], []>, #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @rotate()
  gpu.func @rotate() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [4, 4, 1]>} {
    // CHECK: %[[CST8_I32:.*]] = spirv.Constant 8 : i32
    // CHECK: %[[CST16_I32:.*]] = spirv.Constant 16 : i32
    // CHECK: %[[CST_F32:.*]] = spirv.Constant 4.200000e+01 : f32
    %offset = arith.constant 8 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: spirv.GroupNonUniformRotateKHR <Subgroup>, %[[CST_F32]], %[[CST8_I32]], cluster_size(%[[CST16_I32]])
    %result = gpu.rotate %val, %offset, %width : f32
    gpu.return
  }
}

}
