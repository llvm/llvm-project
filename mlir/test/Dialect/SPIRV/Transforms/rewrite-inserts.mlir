// RUN: mlir-opt -spirv-rewrite-inserts -split-input-file -verify-diagnostics %s -o - | FileCheck %s

spirv.module Logical GLSL450 {
  spirv.func @rewrite(%value0 : f32, %value1 : f32, %value2 : f32, %value3 : i32, %value4: !spirv.array<3xf32>) -> vector<3xf32> "None" {
    %0 = spirv.Undef : vector<3xf32>
    // CHECK: spirv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : (f32, f32, f32) -> vector<3xf32>
    %1 = spirv.CompositeInsert %value0, %0[0 : i32] : f32 into vector<3xf32>
    %2 = spirv.CompositeInsert %value1, %1[1 : i32] : f32 into vector<3xf32>
    %3 = spirv.CompositeInsert %value2, %2[2 : i32] : f32 into vector<3xf32>

    %4 = spirv.Undef : !spirv.array<4xf32>
    // CHECK: spirv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : (f32, f32, f32, f32) -> !spirv.array<4 x f32>
    %5 = spirv.CompositeInsert %value0, %4[0 : i32] : f32 into !spirv.array<4xf32>
    %6 = spirv.CompositeInsert %value1, %5[1 : i32] : f32 into !spirv.array<4xf32>
    %7 = spirv.CompositeInsert %value2, %6[2 : i32] : f32 into !spirv.array<4xf32>
    %8 = spirv.CompositeInsert %value0, %7[3 : i32] : f32 into !spirv.array<4xf32>

    %9 = spirv.Undef : !spirv.struct<(f32, i32, f32)>
    // CHECK: spirv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : (f32, i32, f32) -> !spirv.struct<(f32, i32, f32)>
    %10 = spirv.CompositeInsert %value0, %9[0 : i32] : f32 into !spirv.struct<(f32, i32, f32)>
    %11 = spirv.CompositeInsert %value3, %10[1 : i32] : i32 into !spirv.struct<(f32, i32, f32)>
    %12 = spirv.CompositeInsert %value1, %11[2 : i32] : f32 into !spirv.struct<(f32, i32, f32)>

    %13 = spirv.Undef : !spirv.struct<(f32, !spirv.array<3xf32>)>
    // CHECK: spirv.CompositeConstruct {{%.*}}, {{%.*}} : (f32, !spirv.array<3 x f32>) -> !spirv.struct<(f32, !spirv.array<3 x f32>)>
    %14 = spirv.CompositeInsert %value0, %13[0 : i32] : f32 into !spirv.struct<(f32, !spirv.array<3xf32>)>
    %15 = spirv.CompositeInsert %value4, %14[1 : i32] : !spirv.array<3xf32> into !spirv.struct<(f32, !spirv.array<3xf32>)>

    spirv.ReturnValue %3 : vector<3xf32>
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @insertCoopMatrix(%value : f32) -> !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA> "None" {
    %0 = spirv.Undef : !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>
    // CHECK: spirv.CompositeInsert {{%.*}}, {{%.*}} : f32 into !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>
    %1 = spirv.CompositeInsert %value, %0[0 : i32] : f32 into !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>

    spirv.ReturnValue %1 : !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>
  }
}
