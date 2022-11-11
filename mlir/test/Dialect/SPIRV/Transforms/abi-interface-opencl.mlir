// RUN: mlir-opt -spirv-lower-abi-attrs -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, Addresses], []>, #spirv.resource_limits<>>
} {
  spirv.module Physical64 OpenCL {
    // CHECK-LABEL: spirv.module
    //       CHECK:   spirv.func [[FN:@.*]]({{%.*}}: f32, {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>
    //       CHECK:   spirv.EntryPoint "Kernel" [[FN]]
    //       CHECK:   spirv.ExecutionMode [[FN]] "LocalSize", 32, 1, 1
    spirv.func @kernel(
      %arg0: f32,
      %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>) "None"
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 1, 1]> : vector<3xi32>>} {
      spirv.Return
    }
  }
}
