// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @shuffle_xor()
  gpu.func @shuffle_xor() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
    %mask = arith.constant 8 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[MASK:.+]] = spirv.Constant 8 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.Constant true
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleXor <Subgroup> %[[VAL]], %[[MASK]] : f32, i32
    %result, %valid = gpu.shuffle xor %val, %mask, %width : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, #spirv.resource_limits<subgroup_size = 32>>
} {

gpu.module @kernels {
  gpu.func @shuffle_xor() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
    %mask = arith.constant 8 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // Cannot convert due to shuffle width and target subgroup size mismatch
    // expected-error @+1 {{failed to legalize operation 'gpu.shuffle'}}
    %result, %valid = gpu.shuffle xor %val, %mask, %width : f32
    gpu.return
  }
}

}
