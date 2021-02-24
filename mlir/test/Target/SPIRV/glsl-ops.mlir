// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @fmul(%arg0 : f32, %arg1 : f32, %arg2 : i32) "None" {
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
    // CHECK: {{%.*}} = spv.GLSL.Acos {{%.*}} : f32
    %6 = spv.GLSL.Acos %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Asin {{%.*}} : f32
    %7 = spv.GLSL.Asin %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Atan {{%.*}} : f32
    %8 = spv.GLSL.Atan %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Sinh {{%.*}} : f32
    %9 = spv.GLSL.Sinh %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Cosh {{%.*}} : f32
    %10 = spv.GLSL.Cosh %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.Pow {{%.*}} : f32
    %11 = spv.GLSL.Pow %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spv.GLSL.Round {{%.*}} : f32
    %12 = spv.GLSL.Round %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.FrexpStruct {{%.*}} : f32 -> !spv.struct<(f32, i32)>
    %13 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i32)>
    // CHECK: {{%.*}} = spv.GLSL.Ldexp {{%.*}} : f32, {{%.*}} : i32 -> f32
    %14 = spv.GLSL.Ldexp %arg0 : f32, %arg2 : i32 -> f32
    spv.Return
  }

  spv.func @fclamp(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spv.GLSL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spv.GLSL.FClamp %arg0, %arg1, %arg2 : f32
    spv.Return
  }

  spv.func @uclamp(%arg0 : ui32, %arg1 : ui32, %arg2 : ui32) "None" {
    // CHECK: spv.GLSL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : i32
    %13 = spv.GLSL.UClamp %arg0, %arg1, %arg2 : ui32
    spv.Return
  }

  spv.func @sclamp(%arg0 : si32, %arg1 : si32, %arg2 : si32) "None" {
    // CHECK: spv.GLSL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : si32
    %13 = spv.GLSL.SClamp %arg0, %arg1, %arg2 : si32
    spv.Return
  }

  spv.func @fma(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spv.GLSL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spv.GLSL.Fma %arg0, %arg1, %arg2 : f32
    spv.Return
  }
}
