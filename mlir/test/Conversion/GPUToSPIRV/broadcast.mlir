// RUN: mlir-opt --split-input-file --convert-gpu-to-spirv %s -o - | FileCheck %s

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL: spirv.func @broadcast_specific_lane()
  gpu.func @broadcast_specific_lane() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %lane = arith.constant 0 : i32
    %val = arith.constant 42.0 : f32

    // CHECK: %[[LANE:.+]] = spirv.Constant 0 : i32
    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.GroupNonUniformBroadcast <Subgroup> %[[VAL]], %[[LANE]] : f32, i32
    %result = gpu.subgroup_broadcast %val, specific_lane %lane : f32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL: spirv.func @broadcast_first_active_lane()
  gpu.func @broadcast_first_active_lane() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %val = arith.constant 42.0 : f32

    // CHECK: %[[VAL:.+]] = spirv.Constant 4.200000e+01 : f32
    // CHECK: %{{.+}} = spirv.GroupNonUniformBroadcastFirst <Subgroup> %[[VAL]] : f32
    %result = gpu.subgroup_broadcast %val, first_active_lane : f32
    gpu.return
  }
}

}
