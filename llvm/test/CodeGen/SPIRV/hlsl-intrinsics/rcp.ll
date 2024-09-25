 ; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec2_float_16:]] = OpTypeVector %[[#float_16]] 2
; CHECK-DAG: %[[#vec2_float_32:]] = OpTypeVector %[[#float_32]] 2
; CHECK-DAG: %[[#vec2_float_64:]] = OpTypeVector %[[#float_64]] 2
; CHECK-DAG: %[[#vec3_float_16:]] = OpTypeVector %[[#float_16]] 3
; CHECK-DAG: %[[#vec3_float_32:]] = OpTypeVector %[[#float_32]] 3
; CHECK-DAG: %[[#vec3_float_64:]] = OpTypeVector %[[#float_64]] 3
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_64:]] = OpTypeVector %[[#float_64]] 4
; CHECK-DAG: %[[#const_f64_1:]] = OpConstant %[[#float_64]] 1
; CHECK-DAG: %[[#const_f32_1:]] = OpConstant %[[#float_32]] 1
; CHECK-DAG: %[[#const_f16_1:]] = OpConstant %[[#float_16]] 1

; CHECK-DAG: %[[#vec2_const_ones_f16:]] = OpConstantComposite %[[#vec2_float_16]] %[[#const_f16_1]] %[[#const_f16_1]]
; CHECK-DAG: %[[#vec3_const_ones_f16:]] = OpConstantComposite %[[#vec3_float_16]] %[[#const_f16_1]] %[[#const_f16_1]] %[[#const_f16_1]]
; CHECK-DAG: %[[#vec4_const_ones_f16:]] = OpConstantComposite %[[#vec4_float_16]] %[[#const_f16_1]] %[[#const_f16_1]] %[[#const_f16_1]] %[[#const_f16_1]]

; CHECK-DAG: %[[#vec2_const_ones_f32:]] = OpConstantComposite %[[#vec2_float_32]] %[[#const_f32_1]] %[[#const_f32_1]]
; CHECK-DAG: %[[#vec3_const_ones_f32:]] = OpConstantComposite %[[#vec3_float_32]] %[[#const_f32_1]] %[[#const_f32_1]] %[[#const_f32_1]]
; CHECK-DAG: %[[#vec4_const_ones_f32:]] = OpConstantComposite %[[#vec4_float_32]] %[[#const_f32_1]] %[[#const_f32_1]] %[[#const_f32_1]] %[[#const_f32_1]]

; CHECK-DAG: %[[#vec2_const_ones_f64:]] = OpConstantComposite %[[#vec2_float_64]] %[[#const_f64_1]] %[[#const_f64_1]]
; CHECK-DAG: %[[#vec3_const_ones_f64:]] = OpConstantComposite %[[#vec3_float_64]] %[[#const_f64_1]] %[[#const_f64_1]] %[[#const_f64_1]]
; CHECK-DAG: %[[#vec4_const_ones_f64:]] = OpConstantComposite %[[#vec4_float_64]] %[[#const_f64_1]] %[[#const_f64_1]] %[[#const_f64_1]] %[[#const_f64_1]]


define spir_func noundef half @test_rcp_half(half noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: OpFDiv %[[#float_16]] %[[#const_f16_1]] %[[#arg0]]
  %hlsl.rcp = fdiv half 0xH3C00, %p0
  ret half %hlsl.rcp
}

define spir_func noundef <2 x half> @test_rcp_half2(<2 x half> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_float_16]]
  ; CHECK: OpFDiv %[[#vec2_float_16]] %[[#vec2_const_ones_f16]] %[[#arg0]]
  %hlsl.rcp = fdiv <2 x half> <half 0xH3C00, half 0xH3C00>, %p0
  ret <2 x half> %hlsl.rcp
}

define spir_func noundef <3 x half> @test_rcp_half3(<3 x half> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_16]]
  ; CHECK: OpFDiv %[[#vec3_float_16]] %[[#vec3_const_ones_f16]] %[[#arg0]]
  %hlsl.rcp = fdiv <3 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00>, %p0
  ret <3 x half> %hlsl.rcp
}

define spir_func noundef <4 x half> @test_rcp_half4(<4 x half> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: OpFDiv %[[#vec4_float_16]] %[[#vec4_const_ones_f16]] %[[#arg0]]
  %hlsl.rcp = fdiv <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>, %p0
  ret <4 x half> %hlsl.rcp
}

define spir_func noundef float @test_rcp_float(float noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: OpFDiv %[[#float_32]] %[[#const_f32_1]] %[[#arg0]]
  %hlsl.rcp = fdiv float 1.000000e+00, %p0
  ret float %hlsl.rcp
}

define spir_func noundef <2 x float> @test_rcp_float2(<2 x float> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_float_32]]
  ; CHECK: OpFDiv %[[#vec2_float_32]] %[[#vec2_const_ones_f32]] %[[#arg0]]
  %hlsl.rcp = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %p0
  ret <2 x float> %hlsl.rcp
}

define spir_func noundef <3 x float> @test_rcp_float3(<3 x float> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_32]]
  ; CHECK: OpFDiv %[[#vec3_float_32]] %[[#vec3_const_ones_f32]] %[[#arg0]]
  %hlsl.rcp = fdiv <3 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %p0
  ret <3 x float> %hlsl.rcp
}

define spir_func noundef <4 x float> @test_rcp_float4(<4 x float> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: OpFDiv %[[#vec4_float_32]] %[[#vec4_const_ones_f32]] %[[#arg0]]
  %hlsl.rcp = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %p0
  ret <4 x float> %hlsl.rcp
}

define spir_func noundef double @test_rcp_double(double noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: OpFDiv %[[#float_64]] %[[#const_f64_1]] %[[#arg0]]
  %hlsl.rcp = fdiv double 1.000000e+00, %p0
  ret double %hlsl.rcp
}

define spir_func noundef <2 x double> @test_rcp_double2(<2 x double> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_float_64:]]
  ; CHECK: OpFDiv %[[#vec2_float_64]] %[[#vec2_const_ones_f64]] %[[#arg0]]
  %hlsl.rcp = fdiv <2 x double> <double 1.000000e+00, double 1.000000e+00>, %p0
  ret <2 x double> %hlsl.rcp
}

define spir_func noundef <3 x double> @test_rcp_double3(<3 x double> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_64:]]
  ; CHECK: OpFDiv %[[#vec3_float_64]] %[[#vec3_const_ones_f64]] %[[#arg0]]
  %hlsl.rcp = fdiv <3 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %p0
  ret <3 x double> %hlsl.rcp
}

define spir_func noundef <4 x double> @test_rcp_double4(<4 x double> noundef %p0) #0 {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_64]]
  ; CHECK: OpFDiv %[[#vec4_float_64]] %[[#vec4_const_ones_f64]] %[[#arg0]]
  %hlsl.rcp = fdiv <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %p0
  ret <4 x double> %hlsl.rcp
}
