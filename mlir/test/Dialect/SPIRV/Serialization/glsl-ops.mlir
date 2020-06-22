// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @fmul(%arg0 : f32, %arg1 : f32) "None" {
    // CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %0 = spv.GLSL.Exp %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.FMax {{%.*}}, {{%.*}} : f32
    %1 = spv.GLSL.FMax %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spv.GLSL.Sqrt {{%.*}} : f32
    %2 = spv.GLSL.Sqrt %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Cos {{%.*}} : f32
    %3 = spv.GLSL.Cos %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Sin {{%.*}} : f32
    %4 = spv.GLSL.Sin %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Tan {{%.*}} : f32
    %5 = spv.GLSL.Tan %arg0 : f32
    spv.Return
  }
}
