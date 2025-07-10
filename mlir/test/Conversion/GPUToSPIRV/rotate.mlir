// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformRotateKHR], []>,
    #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @rotate()
  gpu.func @rotate() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %offset = arith.constant 4 : i32
    %width = arith.constant 16 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[OFFSET:.+]] = spirv.Constant 4 : i32
    // CHECK: %[[WIDTH:.+]] = spirv.Constant 16 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.GroupNonUniformRotateKHR <Subgroup> %[[VAL]], %[[OFFSET]], cluster_size(%[[WIDTH]]) : f32, i32, i32 -> f32
    // CHECK: %{{.+}} = spirv.Constant true
    %result, %valid = gpu.rotate %val, %offset, %width : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformRotateKHR], []>,
    #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @rotate_width_less_than_subgroup_size()
  gpu.func @rotate_width_less_than_subgroup_size() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %offset = arith.constant 4 : i32
    %width = arith.constant 8 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[OFFSET:.+]] = spirv.Constant 4 : i32
    // CHECK: %[[WIDTH:.+]] = spirv.Constant 8 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.GroupNonUniformRotateKHR <Subgroup> %[[VAL]], %[[OFFSET]], cluster_size(%[[WIDTH]]) : f32, i32, i32 -> f32
    // CHECK: %[[INVOCATION_ID_ADDR:.+]] = spirv.mlir.addressof @__builtin__SubgroupLocalInvocationId__
    // CHECK: %[[INVOCATION_ID:.+]] = spirv.Load "Input" %[[INVOCATION_ID_ADDR]]
    // CHECK: %{{.+}} = spirv.ULessThan %[[INVOCATION_ID]], %[[WIDTH]]
    %result, %valid = gpu.rotate %val, %offset, %width : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformRotateKHR], []>,
    #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  gpu.func @rotate_with_bigger_than_subgroup_size() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %offset = arith.constant 4 : i32
    %width = arith.constant 32 : i32
    %val = arith.constant 42.0 : f32

    // expected-error @+1 {{failed to legalize operation 'gpu.rotate'}}
    %result, %valid = gpu.rotate %val, %offset, %width : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformRotateKHR], []>,
    #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  gpu.func @rotate_non_const_width(%width: i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %offset = arith.constant 4 : i32
    %val = arith.constant 42.0 : f32

    // expected-error @+1 {{'gpu.rotate' op width is not a constant value}}
    %result, %valid = gpu.rotate %val, %offset, %width : f32
    gpu.return
  }
}

}
