; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; OpSelect: condition and object component counts must match.

; CHECK-DAG: %[[#Bool:]]    = OpTypeBool
; CHECK-DAG: %[[#I32:]]     = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]]     = OpTypeFloat 32
; CHECK-DAG: %[[#I8:]]      = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrI8:]]   = OpTypePointer Function %[[#I8]]
; CHECK-DAG: %[[#V4I32:]]   = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#V4F32:]]   = OpTypeVector %[[#F32]] 4
; CHECK-DAG: %[[#V4Bool:]]  = OpTypeVector %[[#Bool]] 4

; Scalar result, scalar cond.
; CHECK: OpFunction
; CHECK: %[[#SC:]] = OpFunctionParameter %[[#Bool]]
; CHECK: %[[#SA:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#SB:]] = OpFunctionParameter %[[#I32]]
; CHECK: %{{[0-9]+}} = OpSelect %[[#I32]] %[[#SC]] %[[#SA]] %[[#SB]]
define i32 @sel_i32_scond(i1 %c, i32 %a, i32 %b) {
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r
}

; CHECK: OpFunction
; CHECK: %[[#FC:]] = OpFunctionParameter %[[#Bool]]
; CHECK: %[[#FA:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#FB:]] = OpFunctionParameter %[[#F32]]
; CHECK: %{{[0-9]+}} = OpSelect %[[#F32]] %[[#FC]] %[[#FA]] %[[#FB]]
define float @sel_f32_scond(i1 %c, float %a, float %b) {
  %r = select i1 %c, float %a, float %b
  ret float %r
}

; CHECK: OpFunction
; CHECK: %[[#PC:]] = OpFunctionParameter %[[#Bool]]
; CHECK: %[[#PA:]] = OpFunctionParameter %[[#PtrI8]]
; CHECK: %[[#PB:]] = OpFunctionParameter %[[#PtrI8]]
; CHECK: %{{[0-9]+}} = OpSelect %[[#PtrI8]] %[[#PC]] %[[#PA]] %[[#PB]]
define ptr @sel_ptr_scond(i1 %c, ptr %a, ptr %b) {
  %r = select i1 %c, ptr %a, ptr %b
  ret ptr %r
}

; Vector result, scalar (broadcast) cond.
; CHECK: OpFunction
; CHECK: %[[#VSC:]] = OpFunctionParameter %[[#Bool]]
; CHECK: %[[#VSA:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %[[#VSB:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %{{[0-9]+}} = OpSelect %[[#V4I32]] %[[#VSC]] %[[#VSA]] %[[#VSB]]
define <4 x i32> @sel_v4i32_scond(i1 %c, <4 x i32> %a, <4 x i32> %b) {
  %r = select i1 %c, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %r
}

; Vector result, vector cond.
; CHECK: OpFunction
; CHECK: %[[#VVIC:]] = OpFunctionParameter %[[#V4Bool]]
; CHECK: %[[#VVIA:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %[[#VVIB:]] = OpFunctionParameter %[[#V4I32]]
; CHECK: %{{[0-9]+}} = OpSelect %[[#V4I32]] %[[#VVIC]] %[[#VVIA]] %[[#VVIB]]
define <4 x i32> @sel_v4i32_vcond(<4 x i1> %c, <4 x i32> %a, <4 x i32> %b) {
  %r = select <4 x i1> %c, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %r
}

; CHECK: OpFunction
; CHECK: %[[#VVFC:]] = OpFunctionParameter %[[#V4Bool]]
; CHECK: %[[#VVFA:]] = OpFunctionParameter %[[#V4F32]]
; CHECK: %[[#VVFB:]] = OpFunctionParameter %[[#V4F32]]
; CHECK: %{{[0-9]+}} = OpSelect %[[#V4F32]] %[[#VVFC]] %[[#VVFA]] %[[#VVFB]]
define <4 x float> @sel_v4f32_vcond(<4 x i1> %c, <4 x float> %a, <4 x float> %b) {
  %r = select <4 x i1> %c, <4 x float> %a, <4 x float> %b
  ret <4 x float> %r
}

