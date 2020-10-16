// RUN: mlir-opt -spirv-rewrite-inserts -split-input-file -verify-diagnostics %s -o - | FileCheck %s

spv.module Logical GLSL450 {
  spv.func @rewrite(%value0 : f32, %value1 : f32, %value2 : f32, %value3 : i32, %value4: !spv.array<3xf32>) -> vector<3xf32> "None" {
    %0 = spv.undef : vector<3xf32>
    // CHECK: spv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : vector<3xf32>
    %1 = spv.CompositeInsert %value0, %0[0 : i32] : f32 into vector<3xf32>
    %2 = spv.CompositeInsert %value1, %1[1 : i32] : f32 into vector<3xf32>
    %3 = spv.CompositeInsert %value2, %2[2 : i32] : f32 into vector<3xf32>

    %4 = spv.undef : !spv.array<4xf32>
    // CHECK: spv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : !spv.array<4 x f32>
    %5 = spv.CompositeInsert %value0, %4[0 : i32] : f32 into !spv.array<4xf32>
    %6 = spv.CompositeInsert %value1, %5[1 : i32] : f32 into !spv.array<4xf32>
    %7 = spv.CompositeInsert %value2, %6[2 : i32] : f32 into !spv.array<4xf32>
    %8 = spv.CompositeInsert %value0, %7[3 : i32] : f32 into !spv.array<4xf32>

    %9 = spv.undef : !spv.struct<(f32, i32, f32)>
    // CHECK: spv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : !spv.struct<(f32, i32, f32)>
    %10 = spv.CompositeInsert %value0, %9[0 : i32] : f32 into !spv.struct<(f32, i32, f32)>
    %11 = spv.CompositeInsert %value3, %10[1 : i32] : i32 into !spv.struct<(f32, i32, f32)>
    %12 = spv.CompositeInsert %value1, %11[2 : i32] : f32 into !spv.struct<(f32, i32, f32)>

    %13 = spv.undef : !spv.struct<(f32, !spv.array<3xf32>)>
    // CHECK: spv.CompositeConstruct {{%.*}}, {{%.*}} : !spv.struct<(f32, !spv.array<3 x f32>)>
    %14 = spv.CompositeInsert %value0, %13[0 : i32] : f32 into !spv.struct<(f32, !spv.array<3xf32>)>
    %15 = spv.CompositeInsert %value4, %14[1 : i32] : !spv.array<3xf32> into !spv.struct<(f32, !spv.array<3xf32>)>

    spv.ReturnValue %3 : vector<3xf32>
  }
}
