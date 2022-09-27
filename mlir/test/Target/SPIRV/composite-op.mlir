// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @composite_insert(%arg0 : !spirv.struct<(f32, !spirv.struct<(!spirv.array<4xf32>, f32)>)>, %arg1: !spirv.array<4xf32>) -> !spirv.struct<(f32, !spirv.struct<(!spirv.array<4xf32>, f32)>)> "None" {
    // CHECK: spirv.CompositeInsert {{%.*}}, {{%.*}}[1 : i32, 0 : i32] : !spirv.array<4 x f32> into !spirv.struct<(f32, !spirv.struct<(!spirv.array<4 x f32>, f32)>)>
    %0 = spirv.CompositeInsert %arg1, %arg0[1 : i32, 0 : i32] : !spirv.array<4xf32> into !spirv.struct<(f32, !spirv.struct<(!spirv.array<4xf32>, f32)>)>
    spirv.ReturnValue %0: !spirv.struct<(f32, !spirv.struct<(!spirv.array<4xf32>, f32)>)>
  }
  spirv.func @composite_construct_vector(%arg0: f32, %arg1: f32, %arg2 : f32) -> vector<3xf32> "None" {
    // CHECK: spirv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : (f32, f32, f32) -> vector<3xf32>
    %0 = spirv.CompositeConstruct %arg0, %arg1, %arg2 : (f32, f32, f32) -> vector<3xf32>
    spirv.ReturnValue %0: vector<3xf32>
  }
  spirv.func @vector_dynamic_extract(%vec: vector<4xf32>, %id : i32) -> f32 "None" {
    // CHECK: spirv.VectorExtractDynamic %{{.*}}[%{{.*}}] : vector<4xf32>, i32
    %0 = spirv.VectorExtractDynamic %vec[%id] : vector<4xf32>, i32
    spirv.ReturnValue %0: f32
  }
  spirv.func @vector_dynamic_insert(%val: f32, %vec: vector<4xf32>, %id : i32) -> vector<4xf32> "None" {
    // CHECK: spirv.VectorInsertDynamic %{{.*}}, %{{.*}}[%{{.*}}] : vector<4xf32>, i32
    %0 = spirv.VectorInsertDynamic %val, %vec[%id] : vector<4xf32>, i32
    spirv.ReturnValue %0: vector<4xf32>
  }
  spirv.func @vector_shuffle(%vector1: vector<4xf32>, %vector2: vector<2xf32>) -> vector<3xf32> "None" {
    // CHECK: %{{.+}} = spirv.VectorShuffle [1 : i32, 3 : i32, -1 : i32] %{{.+}} : vector<4xf32>, %arg1 : vector<2xf32> -> vector<3xf32>
    %0 = spirv.VectorShuffle [1: i32, 3: i32, 0xffffffff: i32] %vector1: vector<4xf32>, %vector2: vector<2xf32> -> vector<3xf32>
    spirv.ReturnValue %0: vector<3xf32>
  }
}
