// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniformBallot], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL: spirv.func @ballot_i32
  gpu.func @ballot_i32() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
    %c1 = arith.constant 1 : index
    %lane_id = gpu.lane_id
    %pred = arith.cmpi ult, %lane_id, %c1 : index

    // CHECK: %[[VEC:.*]] = spirv.GroupNonUniformBallot <Subgroup> %{{.*}} : vector<4xi32>
    // CHECK: %{{.*}} = spirv.CompositeExtract %[[VEC]][0 : i32] : vector<4xi32>
    %result = gpu.ballot %pred : i32
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, GroupNonUniformBallot, Int64], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL: spirv.func @ballot_i64
  gpu.func @ballot_i64() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
    %c1 = arith.constant 1 : index
    %lane_id = gpu.lane_id
    %pred = arith.cmpi ult, %lane_id, %c1 : index

    // CHECK: %[[VEC:.*]] = spirv.GroupNonUniformBallot <Subgroup> %{{.*}} : vector<4xi32>
    // CHECK: %[[LOW:.*]] = spirv.CompositeExtract %[[VEC]][0 : i32] : vector<4xi32>
    // CHECK: %[[HIGH:.*]] = spirv.CompositeExtract %[[VEC]][1 : i32] : vector<4xi32>
    // CHECK: %[[LOW_EXT:.*]] = spirv.UConvert %[[LOW]] : i32 to i64
    // CHECK: %[[HIGH_EXT:.*]] = spirv.UConvert %[[HIGH]] : i32 to i64
    // CHECK: %[[C32:.*]] = spirv.Constant 32 : i64
    // CHECK: %[[HIGH_SHIFTED:.*]] = spirv.ShiftLeftLogical %[[HIGH_EXT]], %[[C32]] : i64, i64
    // CHECK: %{{.*}} = spirv.BitwiseOr %[[LOW_EXT]], %[[HIGH_SHIFTED]] : i64
    %result = gpu.ballot %pred : i64
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
  gpu.func @ballot_invalid_i8() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
    %c1 = arith.constant 1 : index
    %lane_id = gpu.lane_id
    %pred = arith.cmpi ult, %lane_id, %c1 : index

    // Cannot convert i8 ballot result type
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot'}}
    %result = gpu.ballot %pred : i8
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
  gpu.func @ballot_invalid_i16() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
    %c1 = arith.constant 1 : index
    %lane_id = gpu.lane_id
    %pred = arith.cmpi ult, %lane_id, %c1 : index

    // Cannot convert i16 ballot result type
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot'}}
    %result = gpu.ballot %pred : i16
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
  gpu.func @ballot_invalid_i128() kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
    %c1 = arith.constant 1 : index
    %lane_id = gpu.lane_id
    %pred = arith.cmpi ult, %lane_id, %c1 : index

    // Cannot convert i128 ballot result type
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot'}}
    %result = gpu.ballot %pred : i128
    gpu.return
  }
}

}
