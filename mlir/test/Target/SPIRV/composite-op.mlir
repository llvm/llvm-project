// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @composite_insert(%arg0 : !spv.struct<(f32, !spv.struct<(!spv.array<4xf32>, f32)>)>, %arg1: !spv.array<4xf32>) -> !spv.struct<(f32, !spv.struct<(!spv.array<4xf32>, f32)>)> "None" {
    // CHECK: spv.CompositeInsert {{%.*}}, {{%.*}}[1 : i32, 0 : i32] : !spv.array<4 x f32> into !spv.struct<(f32, !spv.struct<(!spv.array<4 x f32>, f32)>)>
    %0 = spv.CompositeInsert %arg1, %arg0[1 : i32, 0 : i32] : !spv.array<4xf32> into !spv.struct<(f32, !spv.struct<(!spv.array<4xf32>, f32)>)>
    spv.ReturnValue %0: !spv.struct<(f32, !spv.struct<(!spv.array<4xf32>, f32)>)>
  }
  spv.func @composite_construct_vector(%arg0: f32, %arg1: f32, %arg2 : f32) -> vector<3xf32> "None" {
    // CHECK: spv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : vector<3xf32>
    %0 = spv.CompositeConstruct %arg0, %arg1, %arg2 : vector<3xf32>
    spv.ReturnValue %0: vector<3xf32>
  }
  spv.func @vector_dynamic_extract(%vec: vector<4xf32>, %id : i32) -> f32 "None" {
    // CHECK: spv.VectorExtractDynamic %{{.*}}[%{{.*}}] : vector<4xf32>, i32
    %0 = spv.VectorExtractDynamic %vec[%id] : vector<4xf32>, i32
    spv.ReturnValue %0: f32
  }
  spv.func @vector_dynamic_insert(%val: f32, %vec: vector<4xf32>, %id : i32) -> vector<4xf32> "None" {
    // CHECK: spv.VectorInsertDynamic %{{.*}}, %{{.*}}[%{{.*}}] : vector<4xf32>, i32
    %0 = spv.VectorInsertDynamic %val, %vec[%id] : vector<4xf32>, i32
    spv.ReturnValue %0: vector<4xf32>
  }
  spv.func @vector_shuffle(%vector1: vector<4xf32>, %vector2: vector<2xf32>) -> vector<3xf32> "None" {
    // CHECK: %{{.+}} = spv.VectorShuffle [1 : i32, 3 : i32, -1 : i32] %{{.+}} : vector<4xf32>, %arg1 : vector<2xf32> -> vector<3xf32>
    %0 = spv.VectorShuffle [1: i32, 3: i32, 0xffffffff: i32] %vector1: vector<4xf32>, %vector2: vector<2xf32> -> vector<3xf32>
    spv.ReturnValue %0: vector<3xf32>
  }
}
