// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @shuffle_xor()
  gpu.func @shuffle_xor() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
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
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
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

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @shuffle_idx()
  gpu.func @shuffle_idx() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %mask = arith.constant 8 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[MASK:.+]] = spirv.Constant 8 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.Constant true
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffle <Subgroup> %[[VAL]], %[[MASK]] : f32, i32
    %result, %valid = gpu.shuffle idx %val, %mask, %width : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], []>,
    #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @shuffle_down()
  gpu.func @shuffle_down() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %offset = arith.constant 4 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[OFFSET:.+]] = spirv.Constant 4 : i32
    // CHECK: %[[WIDTH:.+]] = spirv.Constant 16 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.Constant true
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleDown <Subgroup> %[[VAL]], %[[OFFSET]] : f32, i32
    %result, %valid = gpu.shuffle down %val, %offset, %width : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], []>,
    #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @shuffle_up()
  gpu.func @shuffle_up() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %offset = arith.constant 4 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[OFFSET:.+]] = spirv.Constant 4 : i32
    // CHECK: %[[WIDTH:.+]] = spirv.Constant 16 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.Constant true
    // CHECK: %{{.+}} = spirv.GroupNonUniformShuffleUp <Subgroup> %[[VAL]], %[[OFFSET]] : f32, i32
    %result, %valid = gpu.shuffle up %val, %offset, %width : f32
    gpu.return
  }
}

}
