// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @math(%arg0 : f32, %arg1 : f32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spirv.GL.Exp {{%.*}} : f32
    %0 = spirv.GL.Exp %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Sqrt {{%.*}} : f32
    %2 = spirv.GL.Sqrt %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Cos {{%.*}} : f32
    %3 = spirv.GL.Cos %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Sin {{%.*}} : f32
    %4 = spirv.GL.Sin %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Tan {{%.*}} : f32
    %5 = spirv.GL.Tan %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Acos {{%.*}} : f32
    %6 = spirv.GL.Acos %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Asin {{%.*}} : f32
    %7 = spirv.GL.Asin %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Atan {{%.*}} : f32
    %8 = spirv.GL.Atan %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Sinh {{%.*}} : f32
    %9 = spirv.GL.Sinh %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Cosh {{%.*}} : f32
    %10 = spirv.GL.Cosh %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Pow {{%.*}} : f32
    %11 = spirv.GL.Pow %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spirv.GL.Round {{%.*}} : f32
    %12 = spirv.GL.Round %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.FrexpStruct {{%.*}} : f32 -> !spirv.struct<(f32, i32)>
    %13 = spirv.GL.FrexpStruct %arg0 : f32 -> !spirv.struct<(f32, i32)>
    // CHECK: {{%.*}} = spirv.GL.Ldexp {{%.*}} : f32, {{%.*}} : i32 -> f32
    %14 = spirv.GL.Ldexp %arg0 : f32, %arg2 : i32 -> f32
    // CHECK: {{%.*}} = spirv.GL.FMix {{%.*}} : f32, {{%.*}} : f32, {{%.*}} : f32 -> f32
    %15 = spirv.GL.FMix %arg0 : f32, %arg1 : f32, %arg0 : f32 -> f32
    spirv.Return
  }

  spirv.func @maxmin(%arg0 : f32, %arg1 : f32, %arg2 : i32, %arg3 : i32) "None" {
    // CHECK: {{%.*}} = spirv.GL.FMax {{%.*}}, {{%.*}} : f32
    %1 = spirv.GL.FMax %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spirv.GL.SMax {{%.*}}, {{%.*}} : i32
    %2 = spirv.GL.SMax %arg2, %arg3 : i32
    // CHECK: {{%.*}} = spirv.GL.UMax {{%.*}}, {{%.*}} : i32
    %3 = spirv.GL.UMax %arg2, %arg3 : i32

    // CHECK: {{%.*}} = spirv.GL.FMin {{%.*}}, {{%.*}} : f32
    %4 = spirv.GL.FMin %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spirv.GL.SMin {{%.*}}, {{%.*}} : i32
    %5 = spirv.GL.SMin %arg2, %arg3 : i32
    // CHECK: {{%.*}} = spirv.GL.UMin {{%.*}}, {{%.*}} : i32
    %6 = spirv.GL.UMin %arg2, %arg3 : i32
    spirv.Return
  }

  spirv.func @fclamp(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spirv.GL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spirv.GL.FClamp %arg0, %arg1, %arg2 : f32
    spirv.Return
  }

  spirv.func @uclamp(%arg0 : ui32, %arg1 : ui32, %arg2 : ui32) "None" {
    // CHECK: spirv.GL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : i32
    %13 = spirv.GL.UClamp %arg0, %arg1, %arg2 : ui32
    spirv.Return
  }

  spirv.func @sclamp(%arg0 : si32, %arg1 : si32, %arg2 : si32) "None" {
    // CHECK: spirv.GL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : si32
    %13 = spirv.GL.SClamp %arg0, %arg1, %arg2 : si32
    spirv.Return
  }

  spirv.func @fma(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spirv.GL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spirv.GL.Fma %arg0, %arg1, %arg2 : f32
    spirv.Return
  }

  spirv.func @findumsb(%arg0 : i32) "None" {
    // CHECK: spirv.GL.FindUMsb {{%.*}} : i32
    %2 = spirv.GL.FindUMsb %arg0 : i32
    spirv.Return
  }
}
