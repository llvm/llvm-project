; RUN: llc  -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext:]] = OpExtInstImport "GLSL.std.450"

; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16

; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32
; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16

; CHECK-DAG: %[[#vec4_float_64:]] = OpTypeVector %[[#float_64]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

; CHECK-DAG: %[[#vec4_int_64:]] = OpTypeVector %[[#int_64]] 4
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec4_int_16:]] = OpTypeVector %[[#int_16]] 4

; CHECK-LABEL: Begin function test_sclamp_i16
define noundef i16 @test_sclamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
  ; CHECK: %[[#i16_arg0:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg1:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg2:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_16]] %[[#op_ext]] SClamp %[[#i16_arg0]] %[[#i16_arg1]] %[[#i16_arg2]]
  %0 = call i16 @llvm.spv.sclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL: Begin function test_sclamp_i32
define noundef i32 @test_sclamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
  ; CHECK: %[[#i32_arg0:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg1:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg2:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_32]] %[[#op_ext]] SClamp %[[#i32_arg0]] %[[#i32_arg1]] %[[#i32_arg2]]
  %0 = call i32 @llvm.spv.sclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_sclamp_i64
define noundef i64 @test_sclamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
  ; CHECK: %[[#i64_arg0:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg1:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg2:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_64]] %[[#op_ext]] SClamp %[[#i64_arg0]] %[[#i64_arg1]] %[[#i64_arg2]]
  %0 = call i64 @llvm.spv.sclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

; CHECK-LABEL: Begin function test_nclamp_half
define noundef half @test_nclamp_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
  ; CHECK: %[[#f16_arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#f16_arg1:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#f16_arg2:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext]] NClamp %[[#f16_arg0]] %[[#f16_arg1]] %[[#f16_arg2]]
  %0 = call half @llvm.spv.nclamp.f16(half %a, half %b, half %c)
  ret half %0
}

; CHECK-LABEL: Begin function test_nclamp_float
define noundef float @test_nclamp_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#f32_arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#f32_arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#f32_arg2:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext]] NClamp %[[#f32_arg0]] %[[#f32_arg1]] %[[#f32_arg2]]
  %0 = call float @llvm.spv.nclamp.f32(float %a, float %b, float %c)
  ret float %0
}

; CHECK-LABEL: Begin function test_nclamp_double
define noundef double @test_nclamp_double(double noundef %a, double noundef %b, double noundef %c) {
entry:
  ; CHECK: %[[#f64_arg0:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#f64_arg1:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#f64_arg2:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_64]] %[[#op_ext]] NClamp %[[#f64_arg0]] %[[#f64_arg1]] %[[#f64_arg2]]
  %0 = call double @llvm.spv.nclamp.f64(double %a, double %b, double %c)
  ret double %0
}

; CHECK-LABEL: Begin function test_uclamp_i16
define noundef i16 @test_uclamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
  ; CHECK: %[[#i16_arg0:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg1:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg2:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_16]] %[[#op_ext]] UClamp %[[#i16_arg0]] %[[#i16_arg1]] %[[#i16_arg2]]
  %0 = call i16 @llvm.spv.uclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL: Begin function test_uclamp_i32
define noundef i32 @test_uclamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
  ; CHECK: %[[#i32_arg0:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg1:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg2:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_32]] %[[#op_ext]] UClamp %[[#i32_arg0]] %[[#i32_arg1]] %[[#i32_arg2]]
  %0 = call i32 @llvm.spv.uclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_uclamp_i64
define noundef i64 @test_uclamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
  ; CHECK: %[[#i64_arg0:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg1:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg2:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_64]] %[[#op_ext]] UClamp %[[#i64_arg0]] %[[#i64_arg1]] %[[#i64_arg2]]
  %0 = call i64 @llvm.spv.uclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

; CHECK-LABEL: Begin function test_sclamp_v4i16
define noundef <4 x i16> @test_sclamp_v4i16(<4 x i16> noundef %a, <4 x i16> noundef %b, <4 x i16> noundef %c) {
entry:
  ; CHECK: %[[#vec4_i16_arg0:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: %[[#vec4_i16_arg1:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: %[[#vec4_i16_arg2:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_int_16]] %[[#op_ext]] SClamp %[[#vec4_i16_arg0]] %[[#vec4_i16_arg1]] %[[#vec4_i16_arg2]]
  %0 = call <4 x i16> @llvm.spv.sclamp.v4i16(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c)
  ret <4 x i16> %0
}

; CHECK-LABEL: Begin function test_sclamp_v4i32
define noundef <4 x i32> @test_sclamp_v4i32(<4 x i32> noundef %a, <4 x i32> noundef %b, <4 x i32> noundef %c) {
entry:
  ; CHECK: %[[#vec4_i32_arg0:]] = OpFunctionParameter %[[#vec4_int_32]]
  ; CHECK: %[[#vec4_i32_arg1:]] = OpFunctionParameter %[[#vec4_int_32]]
  ; CHECK: %[[#vec4_i32_arg2:]] = OpFunctionParameter %[[#vec4_int_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_int_32]] %[[#op_ext]] SClamp %[[#vec4_i32_arg0]] %[[#vec4_i32_arg1]] %[[#vec4_i32_arg2]]
  %0 = call <4 x i32> @llvm.spv.sclamp.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %0
}

; CHECK-LABEL: Begin function test_sclamp_v4i64
define noundef <4 x i64> @test_sclamp_v4i64(<4 x i64> noundef %a, <4 x i64> noundef %b, <4 x i64> noundef %c) {
entry:
  ; CHECK: %[[#vec4_i64_arg0:]] = OpFunctionParameter %[[#vec4_int_64]]
  ; CHECK: %[[#vec4_i64_arg1:]] = OpFunctionParameter %[[#vec4_int_64]]
  ; CHECK: %[[#vec4_i64_arg2:]] = OpFunctionParameter %[[#vec4_int_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_int_64]] %[[#op_ext]] SClamp %[[#vec4_i64_arg0]] %[[#vec4_i64_arg1]] %[[#vec4_i64_arg2]]
  %0 = call <4 x i64> @llvm.spv.sclamp.v4i64(<4 x i64> %a, <4 x i64> %b, <4 x i64> %c)
  ret <4 x i64> %0
}

; CHECK-LABEL: Begin function test_nclamp_v4half
define noundef <4 x half> @test_nclamp_v4half(<4 x half> noundef %a, <4 x half> noundef %b, <4 x half> noundef %c) {
entry:
  ; CHECK: %[[#vec4_f16_arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#vec4_f16_arg1:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#vec4_f16_arg2:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext]] NClamp %[[#vec4_f16_arg0]] %[[#vec4_f16_arg1]] %[[#vec4_f16_arg2]]
  %0 = call <4 x half> @llvm.spv.nclamp.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %0
}

; CHECK-LABEL: Begin function test_nclamp_v4float
define noundef <4 x float> @test_nclamp_v4float(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#vec4_f32_arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#vec4_f32_arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#vec4_f32_arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext]] NClamp %[[#vec4_f32_arg0]] %[[#vec4_f32_arg1]] %[[#vec4_f32_arg2]]
  %0 = call <4 x float> @llvm.spv.nclamp.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %0
}

; CHECK-LABEL: Begin function test_nclamp_v4double
define noundef <4 x double> @test_nclamp_v4double(<4 x double> noundef %a, <4 x double> noundef %b, <4 x double> noundef %c) {
entry:
  ; CHECK: %[[#vec4_f64_arg0:]] = OpFunctionParameter %[[#vec4_float_64]]
  ; CHECK: %[[#vec4_f64_arg1:]] = OpFunctionParameter %[[#vec4_float_64]]
  ; CHECK: %[[#vec4_f64_arg2:]] = OpFunctionParameter %[[#vec4_float_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_64]] %[[#op_ext]] NClamp %[[#vec4_f64_arg0]] %[[#vec4_f64_arg1]] %[[#vec4_f64_arg2]]
  %0 = call <4 x double> @llvm.spv.nclamp.v4f64(<4 x double> %a, <4 x double> %b, <4 x double> %c)
  ret <4 x double> %0
}

; CHECK-LABEL: Begin function test_uclamp_v4i16
define noundef <4 x i16> @test_uclamp_v4i16(<4 x i16> noundef %a, <4 x i16> noundef %b, <4 x i16> noundef %c) {
entry:
  ; CHECK: %[[#vec4_i16_arg0:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: %[[#vec4_i16_arg1:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: %[[#vec4_i16_arg2:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_int_16]] %[[#op_ext]] UClamp %[[#vec4_i16_arg0]] %[[#vec4_i16_arg1]] %[[#vec4_i16_arg2]]
  %0 = call <4 x i16> @llvm.spv.uclamp.v4i16(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c)
  ret <4 x i16> %0
}

; CHECK-LABEL: Begin function test_uclamp_v4i32
define noundef <4 x i32> @test_uclamp_v4i32(<4 x i32> noundef %a, <4 x i32> noundef %b, <4 x i32> noundef %c) {
entry:
  ; CHECK: %[[#vec4_i32_arg0:]] = OpFunctionParameter %[[#vec4_int_32]]
  ; CHECK: %[[#vec4_i32_arg1:]] = OpFunctionParameter %[[#vec4_int_32]]
  ; CHECK: %[[#vec4_i32_arg2:]] = OpFunctionParameter %[[#vec4_int_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_int_32]] %[[#op_ext]] UClamp %[[#vec4_i32_arg0]] %[[#vec4_i32_arg1]] %[[#vec4_i32_arg2]]
  %0 = call <4 x i32> @llvm.spv.uclamp.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %0
}

; CHECK-LABEL: Begin function test_uclamp_v4i64
define noundef <4 x i64> @test_uclamp_v4i64(<4 x i64> noundef %a, <4 x i64> noundef %b, <4 x i64> noundef %c) {
entry:
  ; CHECK: %[[#vec4_i64_arg0:]] = OpFunctionParameter %[[#vec4_int_64]]
  ; CHECK: %[[#vec4_i64_arg1:]] = OpFunctionParameter %[[#vec4_int_64]]
  ; CHECK: %[[#vec4_i64_arg2:]] = OpFunctionParameter %[[#vec4_int_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_int_64]] %[[#op_ext]] UClamp %[[#vec4_i64_arg0]] %[[#vec4_i64_arg1]] %[[#vec4_i64_arg2]]
  %0 = call <4 x i64> @llvm.spv.uclamp.v4i64(<4 x i64> %a, <4 x i64> %b, <4 x i64> %c)
  ret <4 x i64> %0
}

declare half @llvm.spv.nclamp.f16(half, half, half)
declare float @llvm.spv.nclamp.f32(float, float, float)
declare double @llvm.spv.nclamp.f64(double, double, double)
declare i16 @llvm.spv.sclamp.i16(i16, i16, i16)
declare i32 @llvm.spv.sclamp.i32(i32, i32, i32)
declare i64 @llvm.spv.sclamp.i64(i64, i64, i64)
declare i16 @llvm.spv.uclamp.i16(i16, i16, i16)
declare i32 @llvm.spv.uclamp.i32(i32, i32, i32)
declare i64 @llvm.spv.uclamp.i64(i64, i64, i64)
declare <4 x half> @llvm.spv.nclamp.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x float> @llvm.spv.nclamp.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x double> @llvm.spv.nclamp.v4f64(<4 x double>, <4 x double>, <4 x double>)
declare <4 x i16> @llvm.spv.sclamp.v4i16(<4 x i16>, <4 x i16>, <4 x i16>)
declare <4 x i32> @llvm.spv.sclamp.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i64> @llvm.spv.sclamp.v4i64(<4 x i64>, <4 x i64>, <4 x i64>)
declare <4 x i16> @llvm.spv.uclamp.v4i16(<4 x i16>, <4 x i16>, <4 x i16>)
declare <4 x i32> @llvm.spv.uclamp.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i64> @llvm.spv.uclamp.v4i64(<4 x i64>, <4 x i64>, <4 x i64>)


