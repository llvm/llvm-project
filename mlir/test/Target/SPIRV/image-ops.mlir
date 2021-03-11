// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @image(%arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>) "None" {
    // CHECK: {{%.*}} = spv.Image {{%.*}} : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
    %0 = spv.Image %arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
    spv.Return
  }
}
