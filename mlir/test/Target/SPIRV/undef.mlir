// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
    // CHECK: {{%.*}} = spirv.Undef : f32
    // CHECK-NEXT: {{%.*}} = spirv.Undef : f32
    %0 = spirv.Undef : f32
    %1 = spirv.Undef : f32
    %2 = spirv.FAdd %0, %1 : f32
    // CHECK: {{%.*}} = spirv.Undef : vector<4xi32>
    %3 = spirv.Undef : vector<4xi32>
    %4 = spirv.CompositeExtract %3[1 : i32] : vector<4xi32>
    // CHECK: {{%.*}} = spirv.Undef : !spirv.array<4 x !spirv.array<4 x i32>>
    %5 = spirv.Undef : !spirv.array<4x!spirv.array<4xi32>>
    %6 = spirv.CompositeExtract %5[1 : i32, 2 : i32] : !spirv.array<4x!spirv.array<4xi32>>
    // CHECK: {{%.*}} = spirv.Undef : !spirv.ptr<!spirv.struct<(f32)>, StorageBuffer>
    %7 = spirv.Undef : !spirv.ptr<!spirv.struct<(f32)>, StorageBuffer>
    %8 = spirv.Constant 0 : i32
    %9 = spirv.AccessChain %7[%8] : !spirv.ptr<!spirv.struct<(f32)>, StorageBuffer>, i32
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: spirv.func {{@.*}}
  spirv.func @ignore_unused_undef() -> () "None" {
    // CHECK-NEXT: spirv.Return
    %0 = spirv.Undef : f32
    spirv.Return
  }
}
