// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

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
    // CHECK: {{%.*}} = spirv.GL.Fract {{%.*}} : f32
    %16 = spirv.GL.Fract %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Asinh {{%.*}} : f32
    %17 = spirv.GL.Asinh %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Acosh {{%.*}} : f32
    %18 = spirv.GL.Acosh %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Atanh {{%.*}} : f32
    %19 = spirv.GL.Atanh %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Log2 {{%.*}} : f32
    %20 = spirv.GL.Log2 %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Tanh {{%.*}} : f32
    %21 = spirv.GL.Tanh %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Exp2 {{%.*}} : f32
    %22 = spirv.GL.Exp2 %arg0 : f32
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

  spirv.func @findilsb(%arg0 : i32) "None" {
    // CHECK: spirv.GL.FindILsb {{%.*}} : i32
    %2 = spirv.GL.FindILsb %arg0 : i32
    spirv.Return
  }
  spirv.func @findsmsb(%arg0 : i32) "None" {
    // CHECK: spirv.GL.FindSMsb {{%.*}} : i32
    %2 = spirv.GL.FindSMsb %arg0 : i32
    spirv.Return
  }

  spirv.func @findumsb(%arg0 : i32) "None" {
    // CHECK: spirv.GL.FindUMsb {{%.*}} : i32
    %2 = spirv.GL.FindUMsb %arg0 : i32
    spirv.Return
  }

  spirv.func @vector(%arg0 : f32, %arg1 : vector<3xf32>, %arg2 : vector<3xf32>, %arg3: vector<3xi32>) "None" {
    // CHECK: {{%.*}} = spirv.GL.Cross {{%.*}}, {{%.*}} : vector<3xf32>
    %0 = spirv.GL.Cross %arg1, %arg2 : vector<3xf32>
    // CHECK: {{%.*}} = spirv.GL.Normalize {{%.*}} : f32
    %1 = spirv.GL.Normalize %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Normalize {{%.*}} : vector<3xf32>
    %2 = spirv.GL.Normalize %arg1 : vector<3xf32>
    // CHECK: {{%.*}} = spirv.GL.Reflect {{%.*}}, {{%.*}} : f32
    %3 = spirv.GL.Reflect %arg0, %arg0 : f32
    // CHECK: {{%.*}} = spirv.GL.Reflect {{%.*}}, {{%.*}} : vector<3xf32>
    %4 = spirv.GL.Reflect %arg1, %arg2 : vector<3xf32>
    // CHECK: {{%.*}} = spirv.GL.Distance {{%.*}}, {{%.*}} : f32, f32 -> f32
    %5 = spirv.GL.Distance %arg0, %arg0 : f32, f32 -> f32
    // CHECK: {{%.*}} = spirv.GL.Distance {{%.*}}, {{%.*}} : vector<3xf32>, vector<3xf32> -> f32
    %6 = spirv.GL.Distance %arg1, %arg2 : vector<3xf32>, vector<3xf32> -> f32
    // CHECK: {{%.*}} = spirv.GL.FindILsb {{%.*}} : vector<3xi32>
    %7 = spirv.GL.FindILsb %arg3 : vector<3xi32>
    // CHECK: {{%.*}} = spirv.GL.FindSMsb {{%.*}} : vector<3xi32>
    %8 = spirv.GL.FindSMsb %arg3 : vector<3xi32>
    // CHECK: {{%.*}} = spirv.GL.FindUMsb {{%.*}} : vector<3xi32>
    %9 = spirv.GL.FindUMsb %arg3 : vector<3xi32>
    spirv.Return
  }

  spirv.func @pack_half_2x16(%arg0 : i32) "None" {
    // CHECK: {{%.*}} = spirv.GL.UnpackHalf2x16 {{%.*}} : i32 -> vector<2xf32>
    %0 = spirv.GL.UnpackHalf2x16 %arg0 : i32 -> vector<2xf32>
    // CHECK: {{%.*}} = spirv.GL.PackHalf2x16 {{%.*}} : vector<2xf32> -> i32
    %1 = spirv.GL.PackHalf2x16 %0 : vector<2xf32> -> i32
    spirv.Return
  }
}
