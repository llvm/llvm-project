; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"

; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64

; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32
; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64

; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_64:]] = OpTypeVector %[[#float_64]] 4

; CHECK-DAG: %[[#vec4_int_16:]] = OpTypeVector %[[#int_16]] 4
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec4_int_64:]] = OpTypeVector %[[#int_64]] 4


define noundef i32 @sign_half(half noundef %a) {
entry:
; CHECK: %[[#float_16_arg:]] = OpFunctionParameter %[[#float_16]]
; CHECK: %[[#fsign:]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] FSign %[[#float_16_arg]]
; CHECK: %[[#]] = OpConvertFToS %[[#int_32]] %[[#fsign]]
  %elt.sign = call i32 @llvm.spv.sign.f16(half %a)
  ret i32 %elt.sign
}

define noundef i32 @sign_float(float noundef %a) {
entry:
; CHECK: %[[#float_32_arg:]] = OpFunctionParameter %[[#float_32]]
; CHECK: %[[#fsign:]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] FSign %[[#float_32_arg]]
; CHECK: %[[#]] = OpConvertFToS %[[#int_32]] %[[#fsign]]
  %elt.sign = call i32 @llvm.spv.sign.f32(float %a)
  ret i32 %elt.sign
}

define noundef i32 @sign_double(double noundef %a) {
entry:
; CHECK: %[[#float_64_arg:]] = OpFunctionParameter %[[#float_64]]
; CHECK: %[[#fsign:]] = OpExtInst %[[#float_64]] %[[#op_ext_glsl]] FSign %[[#float_64_arg]]
; CHECK: %[[#]] = OpConvertFToS %[[#int_32]] %[[#fsign]]
  %elt.sign = call i32 @llvm.spv.sign.f64(double %a)
  ret i32 %elt.sign
}

define noundef i32 @sign_i16(i16 noundef %a) {
entry:
; CHECK: %[[#int_16_arg:]] = OpFunctionParameter %[[#int_16]]
; CHECK: %[[#ssign:]] = OpExtInst %[[#int_16]] %[[#op_ext_glsl]] SSign %[[#int_16_arg]]
; CHECK: %[[#]] = OpSConvert %[[#int_32]] %[[#ssign]]
  %elt.sign = call i32 @llvm.spv.sign.i16(i16 %a)
  ret i32 %elt.sign
}

define noundef i32 @sign_i32(i32 noundef %a) {
entry:
; CHECK: %[[#int_32_arg:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#]] = OpExtInst %[[#int_32]] %[[#op_ext_glsl]] SSign %[[#int_32_arg]]
  %elt.sign = call i32 @llvm.spv.sign.i32(i32 %a)
  ret i32 %elt.sign
}

define noundef i32 @sign_i64(i64 noundef %a) {
entry:
; CHECK: %[[#int_64_arg:]] = OpFunctionParameter %[[#int_64]]
; CHECK: %[[#ssign:]] = OpExtInst %[[#int_64]] %[[#op_ext_glsl]] SSign %[[#int_64_arg]]
; CHECK: %[[#]] = OpSConvert %[[#int_32]] %[[#ssign]]
  %elt.sign = call i32 @llvm.spv.sign.i64(i64 %a)
  ret i32 %elt.sign
}

define noundef <4 x i32> @sign_half_vector(<4 x half> noundef %a) {
entry:
; CHECK: %[[#vec4_float_16_arg:]] = OpFunctionParameter %[[#vec4_float_16]]
; CHECK: %[[#fsign:]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] FSign %[[#vec4_float_16_arg]]
; CHECK: %[[#]] = OpConvertFToS %[[#vec4_int_32]] %[[#fsign]]
  %elt.sign = call <4 x i32> @llvm.spv.sign.v4f16(<4 x half> %a)
  ret <4 x i32> %elt.sign
}

define noundef <4 x i32> @sign_float_vector(<4 x float> noundef %a) {
entry:
; CHECK: %[[#vec4_float_32_arg:]] = OpFunctionParameter %[[#vec4_float_32]]
; CHECK: %[[#fsign:]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FSign %[[#vec4_float_32_arg]]
; CHECK: %[[#]] = OpConvertFToS %[[#vec4_int_32]] %[[#fsign]]
  %elt.sign = call <4 x i32> @llvm.spv.sign.v4f32(<4 x float> %a)
  ret <4 x i32> %elt.sign
}

define noundef <4 x i32> @sign_double_vector(<4 x double> noundef %a) {
entry:
; CHECK: %[[#vec4_float_64_arg:]] = OpFunctionParameter %[[#vec4_float_64]]
; CHECK: %[[#fsign:]] = OpExtInst %[[#vec4_float_64]] %[[#op_ext_glsl]] FSign %[[#vec4_float_64_arg]]
; CHECK: %[[#]] = OpConvertFToS %[[#vec4_int_32]] %[[#fsign]]
  %elt.sign = call <4 x i32> @llvm.spv.sign.v4f64(<4 x double> %a)
  ret <4 x i32> %elt.sign
}

define noundef <4 x i32> @sign_i16_vector(<4 x i16> noundef %a) {
entry:
; CHECK: %[[#vec4_int_16_arg:]] = OpFunctionParameter %[[#vec4_int_16]]
; CHECK: %[[#ssign:]] = OpExtInst %[[#vec4_int_16]] %[[#op_ext_glsl]] SSign %[[#vec4_int_16_arg]]
; CHECK: %[[#]] = OpSConvert %[[#vec4_int_32]] %[[#ssign]]
  %elt.sign = call <4 x i32> @llvm.spv.sign.v4i16(<4 x i16> %a)
  ret <4 x i32> %elt.sign
}

define noundef <4 x i32> @sign_i32_vector(<4 x i32> noundef %a) {
entry:
; CHECK: %[[#vec4_int_32_arg:]] = OpFunctionParameter %[[#vec4_int_32]]
; CHECK: %[[#]] = OpExtInst %[[#vec4_int_32]] %[[#op_ext_glsl]] SSign %[[#vec4_int_32_arg]]
  %elt.sign = call <4 x i32> @llvm.spv.sign.v4i32(<4 x i32> %a)
  ret <4 x i32> %elt.sign
}

define noundef <4 x i32> @sign_i64_vector(<4 x i64> noundef %a) {
entry:
; CHECK: %[[#vec4_int_64_arg:]] = OpFunctionParameter %[[#vec4_int_64]]
; CHECK: %[[#ssign:]] = OpExtInst %[[#vec4_int_64]] %[[#op_ext_glsl]] SSign %[[#vec4_int_64_arg]]
; CHECK: %[[#]] = OpSConvert %[[#vec4_int_32]] %[[#ssign]]
  %elt.sign = call <4 x i32> @llvm.spv.sign.v4i64(<4 x i64> %a)
  ret <4 x i32> %elt.sign
}

declare i32 @llvm.spv.sign.f16(half)
declare i32 @llvm.spv.sign.f32(float)
declare i32 @llvm.spv.sign.f64(double)

declare i32 @llvm.spv.sign.i16(i16)
declare i32 @llvm.spv.sign.i32(i32)
declare i32 @llvm.spv.sign.i64(i64)

declare <4 x i32> @llvm.spv.sign.v4f16(<4 x half>)
declare <4 x i32> @llvm.spv.sign.v4f32(<4 x float>)
declare <4 x i32> @llvm.spv.sign.v4f64(<4 x double>)

declare <4 x i32> @llvm.spv.sign.v4i16(<4 x i16>)
declare <4 x i32> @llvm.spv.sign.v4i32(<4 x i32>)
declare <4 x i32> @llvm.spv.sign.v4i64(<4 x i64>)
