; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,OCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --check-prefixes=CHECK,VK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; OCL:           [[set:%[0-9]+]] = OpExtInstImport "OpenCL.std"
; VK:            [[set:%[0-9]+]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG:     [[f32:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG:     [[i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG:     [[f64:%[0-9]+]] = OpTypeFloat 64
; CHECK-DAG:     [[i64:%[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG:     [[vecf32:%[0-9]+]] = OpTypeVector [[f32]]
; CHECK-DAG:     [[veci32:%[0-9]+]] = OpTypeVector [[i32]]
; CHECK-DAG:     [[vecf64:%[0-9]+]] = OpTypeVector [[f64]]
; CHECK-DAG:     [[veci64:%[0-9]+]] = OpTypeVector [[i64]]

; The generic lowering expands the intrinsic into G_FRINT + G_FPTOSI, which
; select to OpenCL.std rint / GLSL.std.450 RoundEven followed by OpConvertFToS.
; OCL:        [[rounded_i32_f32:%[0-9]+]] = OpExtInst [[f32]] [[set]] rint %[[#]]
; VK:         [[rounded_i32_f32:%[0-9]+]] = OpExtInst [[f32]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[i32]] [[rounded_i32_f32]]
; OCL:        [[rounded_i32_f64:%[0-9]+]] = OpExtInst [[f64]] [[set]] rint %[[#]]
; VK:         [[rounded_i32_f64:%[0-9]+]] = OpExtInst [[f64]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[i32]] [[rounded_i32_f64]]
; OCL:        [[rounded_i64_f32:%[0-9]+]] = OpExtInst [[f32]] [[set]] rint %[[#]]
; VK:         [[rounded_i64_f32:%[0-9]+]] = OpExtInst [[f32]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[i64]] [[rounded_i64_f32]]
; OCL:        [[rounded_i64_f64:%[0-9]+]] = OpExtInst [[f64]] [[set]] rint %[[#]]
; VK:         [[rounded_i64_f64:%[0-9]+]] = OpExtInst [[f64]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[i64]] [[rounded_i64_f64]]
; OCL:        [[rounded_v4i32_f32:%[0-9]+]] = OpExtInst [[vecf32]] [[set]] rint %[[#]]
; VK:         [[rounded_v4i32_f32:%[0-9]+]] = OpExtInst [[vecf32]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[veci32]] [[rounded_v4i32_f32]]
; OCL:        [[rounded_v4i32_f64:%[0-9]+]] = OpExtInst [[vecf64]] [[set]] rint %[[#]]
; VK:         [[rounded_v4i32_f64:%[0-9]+]] = OpExtInst [[vecf64]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[veci32]] [[rounded_v4i32_f64]]
; OCL:        [[rounded_v4i64_f32:%[0-9]+]] = OpExtInst [[vecf32]] [[set]] rint %[[#]]
; VK:         [[rounded_v4i64_f32:%[0-9]+]] = OpExtInst [[vecf32]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[veci64]] [[rounded_v4i64_f32]]
; OCL:        [[rounded_v4i64_f64:%[0-9]+]] = OpExtInst [[vecf64]] [[set]] rint %[[#]]
; VK:         [[rounded_v4i64_f64:%[0-9]+]] = OpExtInst [[vecf64]] [[set]] RoundEven %[[#]]
; CHECK-NEXT:      %[[#]] = OpConvertFToS [[veci64]] [[rounded_v4i64_f64]]

define spir_func i32 @test_lrint_i32_f32(float %arg0) {
entry:
  %0 = call i32 @llvm.lrint.i32.f32(float %arg0)
  ret i32 %0
}

define spir_func i32 @test_lrint_i32_f64(double %arg0) {
entry:
  %0 = call i32 @llvm.lrint.i32.f64(double %arg0)
  ret i32 %0
}

define spir_func i64 @test_lrint_i64_f32(float %arg0) {
entry:
  %0 = call i64 @llvm.lrint.i64.f32(float %arg0)
  ret i64 %0
}

define spir_func i64 @test_lrint_i64_f64(double %arg0) {
entry:
  %0 = call i64 @llvm.lrint.i64.f64(double %arg0)
  ret i64 %0
}

define spir_func <4 x i32> @test_lrint_v4i32_f32(<4 x float> %arg0) {
entry:
  %0 = call <4 x i32> @llvm.lrint.v4i32.f32(<4 x float> %arg0)
  ret <4 x i32> %0
}

define spir_func <4 x i32> @test_lrint_v4i32_f64(<4 x double> %arg0) {
entry:
  %0 = call <4 x i32> @llvm.lrint.v4i32.f64(<4 x double> %arg0)
  ret <4 x i32> %0
}

define spir_func <4 x i64> @test_lrint_v4i64_f32(<4 x float> %arg0) {
entry:
  %0 = call <4 x i64> @llvm.lrint.v4i64.f32(<4 x float> %arg0)
  ret <4 x i64> %0
}

define spir_func <4 x i64> @test_lrint_v4i64_f64(<4 x double> %arg0) {
entry:
  %0 = call <4 x i64> @llvm.lrint.v4i64.f64(<4 x double> %arg0)
  ret <4 x i64> %0
}

declare i32 @llvm.lrint.i32.f32(float)
declare i32 @llvm.lrint.i32.f64(double)
declare i64 @llvm.lrint.i64.f32(float)
declare i64 @llvm.lrint.i64.f64(double)

declare <4 x i32> @llvm.lrint.v4i32.f32(<4 x float>)
declare <4 x i32> @llvm.lrint.v4i32.f64(<4 x double>)
declare <4 x i64> @llvm.lrint.v4i64.f32(<4 x float>)
declare <4 x i64> @llvm.lrint.v4i64.f64(<4 x double>)
