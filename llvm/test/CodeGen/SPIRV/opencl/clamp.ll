; RUN: llc  -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext:]] = OpExtInstImport "OpenCL.std"

; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16

; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32
; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16

; CHECK-LABEL: Begin function test_sclamp_i16
define noundef i16 @test_sclamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
  ; CHECK: %[[#i16_arg0:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg1:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg2:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_16]] %[[#op_ext]] s_clamp %[[#i16_arg0]] %[[#i16_arg1]] %[[#i16_arg2]]
  %0 = call i16 @llvm.spv.sclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL: Begin function test_sclamp_i32
define noundef i32 @test_sclamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
  ; CHECK: %[[#i32_arg0:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg1:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg2:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_32]] %[[#op_ext]] s_clamp %[[#i32_arg0]] %[[#i32_arg1]] %[[#i32_arg2]]
  %0 = call i32 @llvm.spv.sclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_sclamp_i64
define noundef i64 @test_sclamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
  ; CHECK: %[[#i64_arg0:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg1:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg2:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_64]] %[[#op_ext]] s_clamp %[[#i64_arg0]] %[[#i64_arg1]] %[[#i64_arg2]]
  %0 = call i64 @llvm.spv.sclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

; CHECK-LABEL: Begin function test_fclamp_half
define noundef half @test_fclamp_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
  ; CHECK: %[[#f16_arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#f16_arg1:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#f16_arg2:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext]] fclamp %[[#f16_arg0]] %[[#f16_arg1]] %[[#f16_arg2]]
  %0 = call half @llvm.spv.fclamp.f16(half %a, half %b, half %c)
  ret half %0
}

; CHECK-LABEL: Begin function test_fclamp_float
define noundef float @test_fclamp_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#f32_arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#f32_arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#f32_arg2:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext]] fclamp %[[#f32_arg0]] %[[#f32_arg1]] %[[#f32_arg2]]
  %0 = call float @llvm.spv.fclamp.f32(float %a, float %b, float %c)
  ret float %0
}

; CHECK-LABEL: Begin function test_fclamp_double
define noundef double @test_fclamp_double(double noundef %a, double noundef %b, double noundef %c) {
entry:
  ; CHECK: %[[#f64_arg0:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#f64_arg1:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#f64_arg2:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_64]] %[[#op_ext]] fclamp %[[#f64_arg0]] %[[#f64_arg1]] %[[#f64_arg2]]
  %0 = call double @llvm.spv.fclamp.f64(double %a, double %b, double %c)
  ret double %0
}

; CHECK-LABEL: Begin function test_uclamp_i16
define noundef i16 @test_uclamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
  ; CHECK: %[[#i16_arg0:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg1:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#i16_arg2:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_16]] %[[#op_ext]] u_clamp %[[#i16_arg0]] %[[#i16_arg1]] %[[#i16_arg2]]
  %0 = call i16 @llvm.spv.uclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL: Begin function test_uclamp_i32
define noundef i32 @test_uclamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
  ; CHECK: %[[#i32_arg0:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg1:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#i32_arg2:]] = OpFunctionParameter %[[#int_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_32]] %[[#op_ext]] u_clamp %[[#i32_arg0]] %[[#i32_arg1]] %[[#i32_arg2]]
  %0 = call i32 @llvm.spv.uclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_uclamp_i64
define noundef i64 @test_uclamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
  ; CHECK: %[[#i64_arg0:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg1:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#i64_arg2:]] = OpFunctionParameter %[[#int_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#int_64]] %[[#op_ext]] u_clamp %[[#i64_arg0]] %[[#i64_arg1]] %[[#i64_arg2]]
  %0 = call i64 @llvm.spv.uclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

declare half @llvm.spv.fclamp.f16(half, half, half)
declare float @llvm.spv.fclamp.f32(float, float, float)
declare double @llvm.spv.fclamp.f64(double, double, double)
declare i16 @llvm.spv.sclamp.i16(i16, i16, i16)
declare i32 @llvm.spv.sclamp.i32(i32, i32, i32)
declare i64 @llvm.spv.sclamp.i64(i64, i64, i64)
declare i16 @llvm.spv.uclamp.i16(i16, i16, i16)
declare i32 @llvm.spv.uclamp.i32(i32, i32, i32)
declare i64 @llvm.spv.uclamp.i64(i64, i64, i64)

