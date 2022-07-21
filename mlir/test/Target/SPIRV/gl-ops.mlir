// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @math(%arg0 : f32, %arg1 : f32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spv.GL.Exp {{%.*}} : f32
    %0 = spv.GL.Exp %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Sqrt {{%.*}} : f32
    %2 = spv.GL.Sqrt %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Cos {{%.*}} : f32
    %3 = spv.GL.Cos %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Sin {{%.*}} : f32
    %4 = spv.GL.Sin %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Tan {{%.*}} : f32
    %5 = spv.GL.Tan %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Acos {{%.*}} : f32
    %6 = spv.GL.Acos %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Asin {{%.*}} : f32
    %7 = spv.GL.Asin %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Atan {{%.*}} : f32
    %8 = spv.GL.Atan %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Sinh {{%.*}} : f32
    %9 = spv.GL.Sinh %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Cosh {{%.*}} : f32
    %10 = spv.GL.Cosh %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.Pow {{%.*}} : f32
    %11 = spv.GL.Pow %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spv.GL.Round {{%.*}} : f32
    %12 = spv.GL.Round %arg0 : f32
    // CHECK: {{%.*}} = spv.GL.FrexpStruct {{%.*}} : f32 -> !spv.struct<(f32, i32)>
    %13 = spv.GL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i32)>
    // CHECK: {{%.*}} = spv.GL.Ldexp {{%.*}} : f32, {{%.*}} : i32 -> f32
    %14 = spv.GL.Ldexp %arg0 : f32, %arg2 : i32 -> f32
    // CHECK: {{%.*}} = spv.GL.FMix {{%.*}} : f32, {{%.*}} : f32, {{%.*}} : f32 -> f32
    %15 = spv.GL.FMix %arg0 : f32, %arg1 : f32, %arg0 : f32 -> f32
    spv.Return
  }

  spv.func @maxmin(%arg0 : f32, %arg1 : f32, %arg2 : i32, %arg3 : i32) "None" {
    // CHECK: {{%.*}} = spv.GL.FMax {{%.*}}, {{%.*}} : f32
    %1 = spv.GL.FMax %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spv.GL.SMax {{%.*}}, {{%.*}} : i32
    %2 = spv.GL.SMax %arg2, %arg3 : i32
    // CHECK: {{%.*}} = spv.GL.UMax {{%.*}}, {{%.*}} : i32
    %3 = spv.GL.UMax %arg2, %arg3 : i32

    // CHECK: {{%.*}} = spv.GL.FMin {{%.*}}, {{%.*}} : f32
    %4 = spv.GL.FMin %arg0, %arg1 : f32
    // CHECK: {{%.*}} = spv.GL.SMin {{%.*}}, {{%.*}} : i32
    %5 = spv.GL.SMin %arg2, %arg3 : i32
    // CHECK: {{%.*}} = spv.GL.UMin {{%.*}}, {{%.*}} : i32
    %6 = spv.GL.UMin %arg2, %arg3 : i32
    spv.Return
  }

  spv.func @fclamp(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spv.GL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spv.GL.FClamp %arg0, %arg1, %arg2 : f32
    spv.Return
  }

  spv.func @uclamp(%arg0 : ui32, %arg1 : ui32, %arg2 : ui32) "None" {
    // CHECK: spv.GL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : i32
    %13 = spv.GL.UClamp %arg0, %arg1, %arg2 : ui32
    spv.Return
  }

  spv.func @sclamp(%arg0 : si32, %arg1 : si32, %arg2 : si32) "None" {
    // CHECK: spv.GL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : si32
    %13 = spv.GL.SClamp %arg0, %arg1, %arg2 : si32
    spv.Return
  }

  spv.func @fma(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spv.GL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spv.GL.Fma %arg0, %arg1, %arg2 : f32
    spv.Return
  }

  spv.func @findumsb(%arg0 : i32) "None" {
    // CHECK: spv.GL.FindUMsb {{%.*}} : i32
    %2 = spv.GL.FindUMsb %arg0 : i32
    spv.Return
  }
}
