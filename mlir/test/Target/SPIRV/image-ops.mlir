// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @image(%arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) "None" {
    // CHECK: {{%.*}} = spv.Image {{%.*}} : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
    %0 = spv.Image %arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
    // CHECK: {{%.*}} = spv.ImageDrefGather {{%.*}} : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,  {{%.*}} : vector<4xf32>,  {{%.*}} : f32 -> vector<4xf32>
    %1 = spv.ImageDrefGather %arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<4xf32>
    spv.Return
  }
  spv.func @image_query_size(%arg0 : !spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>) "None" {
    // CHECK:  {{%.*}} = spv.ImageQuerySize %arg0 : !spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
    %0 = spv.ImageQuerySize %arg0 : !spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
    spv.Return
  }
}
