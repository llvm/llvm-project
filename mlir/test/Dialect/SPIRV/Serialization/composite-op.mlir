// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @composite_insert(%arg0 : !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>, %arg1: !spv.array<4xf32>) -> !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>> {
    // CHECK: spv.CompositeInsert {{%.*}}, {{%.*}}[1 : i32, 0 : i32] : !spv.array<4 x f32> into !spv.struct<f32, !spv.struct<!spv.array<4 x f32>, f32>>
    %0 = spv.CompositeInsert %arg1, %arg0[1 : i32, 0 : i32] : !spv.array<4xf32> into !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>
    spv.ReturnValue %0: !spv.struct<f32, !spv.struct<!spv.array<4xf32>, f32>>
  }
  func @composite_construct_vector(%arg0: f32, %arg1: f32, %arg2 : f32) -> vector<3xf32> {
    // CHECK: spv.CompositeConstruct {{%.*}}, {{%.*}}, {{%.*}} : vector<3xf32>
    %0 = spv.CompositeConstruct %arg0, %arg1, %arg2 : vector<3xf32>
    spv.ReturnValue %0: vector<3xf32>
  }
}
