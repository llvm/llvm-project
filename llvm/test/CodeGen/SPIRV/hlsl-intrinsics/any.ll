; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-HLSL
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-OCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; Make sure spirv operation function calls for any are generated.

; CHECK-HLSL-DAG: OpMemoryModel Logical GLSL450
; CHECK-OCL-DAG: OpMemoryModel Physical32 OpenCL
; CHECK-DAG: OpName %[[#any_bool_arg:]] "a"
; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG: %[[#vec4_16:]] = OpTypeVector %[[#int_16]] 4
; CHECK-DAG: %[[#vec4_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec4_64:]] = OpTypeVector %[[#int_64]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_64:]] = OpTypeVector %[[#float_64]] 4

; CHECK-HLSL-DAG: %[[#const_i64_0:]] = OpConstant %[[#int_64]] 0
; CHECK-HLSL-DAG: %[[#const_i32_0:]] = OpConstant %[[#int_32]] 0
; CHECK-HLSL-DAG: %[[#const_i16_0:]] = OpConstant %[[#int_16]] 0
; CHECK-HLSL-DAG: %[[#const_f64_0:]] = OpConstant %[[#float_64]] 0
; CHECK-HLSL-DAG: %[[#const_f32_0:]] = OpConstant %[[#float_32]] 0
; CHECK-HLSL-DAG: %[[#const_f16_0:]] = OpConstant %[[#float_16]] 0
; CHECK-HLSL-DAG: %[[#vec4_const_zeros_i16:]] = OpConstantComposite %[[#vec4_16]] %[[#const_i16_0]] %[[#const_i16_0]] %[[#const_i16_0]] %[[#const_i16_0]]
; CHECK-HLSL-DAG: %[[#vec4_const_zeros_i32:]] = OpConstantComposite %[[#vec4_32]] %[[#const_i32_0]] %[[#const_i32_0]] %[[#const_i32_0]] %[[#const_i32_0]]
; CHECK-HLSL-DAG: %[[#vec4_const_zeros_i64:]] = OpConstantComposite %[[#vec4_64]] %[[#const_i64_0]] %[[#const_i64_0]] %[[#const_i64_0]] %[[#const_i64_0]]
; CHECK-HLSL-DAG: %[[#vec4_const_zeros_f16:]] = OpConstantComposite %[[#vec4_float_16]] %[[#const_f16_0]] %[[#const_f16_0]] %[[#const_f16_0]] %[[#const_f16_0]]
; CHECK-HLSL-DAG: %[[#vec4_const_zeros_f32:]] = OpConstantComposite %[[#vec4_float_32]] %[[#const_f32_0]] %[[#const_f32_0]] %[[#const_f32_0]] %[[#const_f32_0]]
; CHECK-HLSL-DAG: %[[#vec4_const_zeros_f64:]] = OpConstantComposite %[[#vec4_float_64]] %[[#const_f64_0]] %[[#const_f64_0]] %[[#const_f64_0]] %[[#const_f64_0]]

; CHECK-OCL-DAG: %[[#const_i64_0:]] = OpConstantNull %[[#int_64]]
; CHECK-OCL-DAG: %[[#const_i32_0:]] = OpConstantNull %[[#int_32]]
; CHECK-OCL-DAG: %[[#const_i16_0:]] = OpConstantNull %[[#int_16]]
; CHECK-OCL-DAG: %[[#const_f64_0:]] = OpConstantNull %[[#float_64]] 
; CHECK-OCL-DAG: %[[#const_f32_0:]] = OpConstantNull %[[#float_32]]
; CHECK-OCL-DAG: %[[#const_f16_0:]] = OpConstantNull %[[#float_16]]
; CHECK-OCL-DAG: %[[#vec4_const_zeros_i16:]] = OpConstantNull %[[#vec4_16]]
; CHECK-OCL-DAG: %[[#vec4_const_zeros_i32:]] = OpConstantNull %[[#vec4_32]]
; CHECK-OCL-DAG: %[[#vec4_const_zeros_i64:]] = OpConstantNull %[[#vec4_64]]
; CHECK-OCL-DAG: %[[#vec4_const_zeros_f16:]] = OpConstantNull %[[#vec4_float_16]]
; CHECK-OCL-DAG: %[[#vec4_const_zeros_f32:]] = OpConstantNull %[[#vec4_float_32]]
; CHECK-OCL-DAG: %[[#vec4_const_zeros_f64:]] = OpConstantNull %[[#vec4_float_64]]

define noundef i1 @any_int64_t(i64 noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpINotEqual %[[#bool]] %[[#arg0]] %[[#const_i64_0]]
  %hlsl.any = call i1 @llvm.spv.any.i64(i64 %p0)
  ret i1 %hlsl.any
}


define noundef i1 @any_int(i32 noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpINotEqual %[[#bool]] %[[#arg0]] %[[#const_i32_0]]
  %hlsl.any = call i1 @llvm.spv.any.i32(i32 %p0)
  ret i1 %hlsl.any
}


define noundef i1 @any_int16_t(i16 noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpINotEqual %[[#bool]] %[[#arg0]] %[[#const_i16_0]]
  %hlsl.any = call i1 @llvm.spv.any.i16(i16 %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_double(double noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpFOrdNotEqual %[[#bool]] %[[#arg0]] %[[#const_f64_0]]
  %hlsl.any = call i1 @llvm.spv.any.f64(double %p0)
  ret i1 %hlsl.any
}


define noundef i1 @any_float(float noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpFOrdNotEqual %[[#bool]] %[[#arg0]] %[[#const_f32_0]]
  %hlsl.any = call i1 @llvm.spv.any.f32(float %p0)
  ret i1 %hlsl.any
}


define noundef i1 @any_half(half noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpFOrdNotEqual %[[#bool]] %[[#arg0]] %[[#const_f16_0]]
  %hlsl.any = call i1 @llvm.spv.any.f16(half %p0)
  ret i1 %hlsl.any
}


define noundef i1 @any_bool4(<4 x i1> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_bool]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#arg0]]
  %hlsl.any = call i1 @llvm.spv.any.v4i1(<4 x i1> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_short4(<4 x i16> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#shortVecNotEq:]] = OpINotEqual %[[#vec4_bool]] %[[#arg0]] %[[#vec4_const_zeros_i16]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#shortVecNotEq]]
  %hlsl.any = call i1 @llvm.spv.any.v4i16(<4 x i16> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_int4(<4 x i32> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#i32VecNotEq:]] = OpINotEqual %[[#vec4_bool]] %[[#arg0]] %[[#vec4_const_zeros_i32]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#i32VecNotEq]]
  %hlsl.any = call i1 @llvm.spv.any.v4i32(<4 x i32> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_int64_t4(<4 x i64> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#i64VecNotEq:]] = OpINotEqual %[[#vec4_bool]] %[[#arg0]] %[[#vec4_const_zeros_i64]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#i64VecNotEq]]
  %hlsl.any = call i1 @llvm.spv.any.v4i64(<4 x i64> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_half4(<4 x half> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#f16VecNotEq:]] = OpFOrdNotEqual %[[#vec4_bool]] %[[#arg0]] %[[#vec4_const_zeros_f16]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#f16VecNotEq]]
  %hlsl.any = call i1 @llvm.spv.any.v4f16(<4 x half> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_float4(<4 x float> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#f32VecNotEq:]] = OpFOrdNotEqual %[[#vec4_bool]] %[[#arg0]] %[[#vec4_const_zeros_f32]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#f32VecNotEq]]
  %hlsl.any = call i1 @llvm.spv.any.v4f32(<4 x float> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_double4(<4 x double> noundef %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#f64VecNotEq:]] = OpFOrdNotEqual %[[#vec4_bool]] %[[#arg0]] %[[#vec4_const_zeros_f64]]
  ; CHECK: %[[#]] = OpAny %[[#bool]] %[[#f64VecNotEq]]
  %hlsl.any = call i1 @llvm.spv.any.v4f64(<4 x double> %p0)
  ret i1 %hlsl.any
}

define noundef i1 @any_bool(i1 noundef %a) {
entry:
  ; CHECK: %[[#any_bool_arg:]] = OpFunctionParameter %[[#bool]]
  ; CHECK: OpReturnValue %[[#any_bool_arg]]
  %hlsl.any = call i1 @llvm.spv.any.i1(i1 %a)
  ret i1 %hlsl.any
}

declare i1 @llvm.spv.any.v4f16(<4 x half>)
declare i1 @llvm.spv.any.v4f32(<4 x float>)
declare i1 @llvm.spv.any.v4f64(<4 x double>)
declare i1 @llvm.spv.any.v4i1(<4 x i1>)
declare i1 @llvm.spv.any.v4i16(<4 x i16>)
declare i1 @llvm.spv.any.v4i32(<4 x i32>)
declare i1 @llvm.spv.any.v4i64(<4 x i64>)
declare i1 @llvm.spv.any.i1(i1)
declare i1 @llvm.spv.any.i16(i16)
declare i1 @llvm.spv.any.i32(i32)
declare i1 @llvm.spv.any.i64(i64)
declare i1 @llvm.spv.any.f16(half)
declare i1 @llvm.spv.any.f32(float)
declare i1 @llvm.spv.any.f64(double)
