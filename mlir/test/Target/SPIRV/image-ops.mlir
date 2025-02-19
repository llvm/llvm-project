// RUN: mlir-translate --no-implicit-module --split-input-file --test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, ImageQuery], []> {
  spirv.func @image(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) "None" {
    // CHECK: {{%.*}} = spirv.Image {{%.*}} : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
    %0 = spirv.Image %arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
    // CHECK: {{%.*}} = spirv.ImageDrefGather {{%.*}} : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,  {{%.*}} : vector<4xf32>,  {{%.*}} : f32 -> vector<4xf32>
    %1 = spirv.ImageDrefGather %arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<4xf32>
    spirv.Return
  }
  spirv.func @image_query_size(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>) "None" {
    // CHECK:  {{%.*}} = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
    %0 = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
    spirv.Return
  }
  spirv.func @image_write(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>) "None" {
    // CHECK:  spirv.ImageWrite %arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>
    spirv.ImageWrite %arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>
    spirv.Return
  }
  spirv.func @main() "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, StorageImageWriteWithoutFormat], []> {
  spirv.func @image_write(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>) "None" {
    // CHECK:  spirv.ImageWrite %arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>
    spirv.ImageWrite %arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>
    spirv.Return
  }
  spirv.func @main() "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}
