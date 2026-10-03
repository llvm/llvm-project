; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CL
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CL
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --check-prefixes=CHECK,VK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; llvm.minimum/llvm.maximum propagate NaNs and order -0.0 before +0.0, whereas
; OpenCL.std fmin/fmax and GLSL.std.450 NMin/NMax return the non-NaN operand and
; may return either zero. Check that the missing NaN propagation and signed-zero
; handling are added, and dropped again under nnan and nsz.

; CL-DAG: %[[#Ext:]] = OpExtInstImport "OpenCL.std"
; VK-DAG: %[[#Ext:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#False:]] = OpConstantFalse %[[#Bool]]
; CHECK-DAG: %[[#F16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#F64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V2F32:]] = OpTypeVector %[[#F32]] 2
; CHECK-DAG: %[[#V3F32:]] = OpTypeVector %[[#F32]] 3
; CHECK-DAG: %[[#V4F32:]] = OpTypeVector %[[#F32]] 4
; CHECK-DAG: %[[#V4Bool:]] = OpTypeVector %[[#Bool]] 4
; CHECK-DAG: %[[#NaN16:]] = OpConstant %[[#F16]] 32256
; CHECK-DAG: %[[#NaN32:]] = OpConstant %[[#F32]] 0x1.8p+128
; CHECK-DAG: %[[#NaN64:]] = OpConstant %[[#F64]] 0x1.8p+1024
; CL-DAG: %[[#Zero32:]] = OpConstantNull %[[#F32]]
; VK-DAG: %[[#Zero32:]] = OpConstant %[[#F32]] 0{{$}}
; CHECK-DAG: %[[#NegZeroBits32:]] = OpConstant %[[#I32]] 2147483648
; CL-DAG: %[[#PosZeroBits32:]] = OpConstantNull %[[#I32]]
; VK-DAG: %[[#PosZeroBits32:]] = OpConstant %[[#I32]] 0{{$}}

; CHECK-LABEL: Begin function minimum_f32
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#Min:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; VK: %[[#Min:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; CL: %[[#Ord:]] = OpOrdered %[[#Bool]] %[[#A]] %[[#B]]
; VK: %[[#NanA:]] = OpIsNan %[[#Bool]] %[[#A]]
; VK: %[[#NanB:]] = OpIsNan %[[#Bool]] %[[#B]]
; VK: %[[#Uno:]] = OpLogicalOr %[[#Bool]] %[[#NanA]] %[[#NanB]]
; VK: %[[#Ord:]] = OpLogicalNot %[[#Bool]] %[[#Uno]]
; CHECK: %[[#NaNSel:]] = OpSelect %[[#F32]] %[[#Ord]] %[[#Min]] %[[#NaN32]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#Bool]] %[[#NaNSel]] %[[#Zero32]]
; CHECK: %[[#ABits:]] = OpBitcast %[[#I32]] %[[#A]]
; CHECK: %[[#ANegZero:]] = OpIEqual %[[#Bool]] %[[#ABits]] %[[#NegZeroBits32]]
; CHECK: %[[#ACond:]] = OpLogicalOr %[[#Bool]] %[[#False]] %[[#ANegZero]]
; CHECK: %[[#ASel:]] = OpSelect %[[#F32]] %[[#ACond]] %[[#A]] %[[#NaNSel]]
; CHECK: %[[#BBits:]] = OpBitcast %[[#I32]] %[[#B]]
; CHECK: %[[#BNegZero:]] = OpIEqual %[[#Bool]] %[[#BBits]] %[[#NegZeroBits32]]
; CHECK: %[[#BCond:]] = OpLogicalOr %[[#Bool]] %[[#False]] %[[#BNegZero]]
; CHECK: %[[#BSel:]] = OpSelect %[[#F32]] %[[#BCond]] %[[#B]] %[[#ASel]]
; CHECK: %[[#Res:]] = OpSelect %[[#F32]] %[[#IsZero]] %[[#BSel]] %[[#NaNSel]]
; CHECK: OpReturnValue %[[#Res]]
define float @minimum_f32(float %a, float %b) {
  %r = call float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}

; CHECK-LABEL: Begin function maximum_f32
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#Max:]] = OpExtInst %[[#F32]] %[[#Ext]] fmax %[[#A]] %[[#B]]
; VK: %[[#Max:]] = OpExtInst %[[#F32]] %[[#Ext]] NMax %[[#A]] %[[#B]]
; CL: %[[#Ord:]] = OpOrdered %[[#Bool]] %[[#A]] %[[#B]]
; VK: %[[#Ord:]] = OpLogicalNot %[[#Bool]]
; CHECK: %[[#NaNSel:]] = OpSelect %[[#F32]] %[[#Ord]] %[[#Max]] %[[#NaN32]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#Bool]] %[[#NaNSel]] %[[#Zero32]]
; CHECK: %[[#ABits:]] = OpBitcast %[[#I32]] %[[#A]]
; CHECK: %[[#APosZero:]] = OpIEqual %[[#Bool]] %[[#ABits]] %[[#PosZeroBits32]]
; CHECK: %[[#ACond:]] = OpLogicalOr %[[#Bool]] %[[#False]] %[[#APosZero]]
; CHECK: %[[#ASel:]] = OpSelect %[[#F32]] %[[#ACond]] %[[#A]] %[[#NaNSel]]
; CHECK: %[[#BBits:]] = OpBitcast %[[#I32]] %[[#B]]
; CHECK: %[[#BPosZero:]] = OpIEqual %[[#Bool]] %[[#BBits]] %[[#PosZeroBits32]]
; CHECK: %[[#BCond:]] = OpLogicalOr %[[#Bool]] %[[#False]] %[[#BPosZero]]
; CHECK: %[[#BSel:]] = OpSelect %[[#F32]] %[[#BCond]] %[[#B]] %[[#ASel]]
; CHECK: %[[#Res:]] = OpSelect %[[#F32]] %[[#IsZero]] %[[#BSel]] %[[#NaNSel]]
; CHECK: OpReturnValue %[[#Res]]
define float @maximum_f32(float %a, float %b) {
  %r = call float @llvm.maximum.f32(float %a, float %b)
  ret float %r
}

; With nnan, only the signed-zero handling remains.
; CHECK-LABEL: Begin function minimum_f32_nnan
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#Min:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; VK: %[[#Min:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; CHECK-NOT: OpOrdered
; CHECK-NOT: OpIsNan
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#Bool]] %[[#Min]] %[[#Zero32]]
; CHECK: %[[#Res:]] = OpSelect %[[#F32]] %[[#IsZero]] %[[#]] %[[#Min]]
; CHECK: OpReturnValue %[[#Res]]
define float @minimum_f32_nnan(float %a, float %b) {
  %r = call nnan float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}

; With nsz, only the NaN propagation remains.
; CHECK-LABEL: Begin function minimum_f32_nsz
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#Min:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; VK: %[[#Min:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; CL: %[[#Ord:]] = OpOrdered %[[#Bool]] %[[#A]] %[[#B]]
; VK: %[[#Ord:]] = OpLogicalNot %[[#Bool]]
; CHECK: %[[#Res:]] = OpSelect %[[#F32]] %[[#Ord]] %[[#Min]] %[[#NaN32]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define float @minimum_f32_nsz(float %a, float %b) {
  %r = call nsz float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}

; With nnan and nsz, fmin/fmax (NMin/NMax) is enough.
; CHECK-LABEL: Begin function minimum_f32_nnan_nsz
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CHECK-NEXT: OpLabel
; CL-NEXT: %[[#Res:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; VK-NEXT: %[[#Res:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define float @minimum_f32_nnan_nsz(float %a, float %b) {
  %r = call nnan nsz float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}

; CHECK-LABEL: Begin function maximum_f32_nnan_nsz
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CHECK-NEXT: OpLabel
; CL-NEXT: %[[#Res:]] = OpExtInst %[[#F32]] %[[#Ext]] fmax %[[#A]] %[[#B]]
; VK-NEXT: %[[#Res:]] = OpExtInst %[[#F32]] %[[#Ext]] NMax %[[#A]] %[[#B]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define float @maximum_f32_nnan_nsz(float %a, float %b) {
  %r = call fast float @llvm.maximum.f32(float %a, float %b)
  ret float %r
}

; CHECK-LABEL: Begin function minimum_f16
; CL: %[[#Min:]] = OpExtInst %[[#F16]] %[[#Ext]] fmin
; VK: %[[#Min:]] = OpExtInst %[[#F16]] %[[#Ext]] NMin
; CHECK: %[[#NaNSel:]] = OpSelect %[[#F16]] %[[#]] %[[#Min]] %[[#NaN16]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#]] %[[#NaNSel]] %[[#]]
; CHECK: %[[#Res:]] = OpSelect %[[#F16]] %[[#IsZero]] %[[#]] %[[#NaNSel]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define half @minimum_f16(half %a, half %b) {
  %r = call half @llvm.minimum.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: Begin function maximum_f64
; CL: %[[#Max:]] = OpExtInst %[[#F64]] %[[#Ext]] fmax
; VK: %[[#Max:]] = OpExtInst %[[#F64]] %[[#Ext]] NMax
; CHECK: %[[#NaNSel:]] = OpSelect %[[#F64]] %[[#]] %[[#Max]] %[[#NaN64]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#]] %[[#NaNSel]] %[[#]]
; CHECK: %[[#Res:]] = OpSelect %[[#F64]] %[[#IsZero]] %[[#]] %[[#NaNSel]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define double @maximum_f64(double %a, double %b) {
  %r = call double @llvm.maximum.f64(double %a, double %b)
  ret double %r
}

; CHECK-LABEL: Begin function minimum_v2f32
; CL: %[[#Min:]] = OpExtInst %[[#V2F32]] %[[#Ext]] fmin
; VK: %[[#Min:]] = OpExtInst %[[#V2F32]] %[[#Ext]] NMin
; CHECK: %[[#NaNSel:]] = OpSelect %[[#V2F32]] %[[#]] %[[#Min]] %[[#]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#]] %[[#NaNSel]] %[[#]]
; CHECK: %[[#Res:]] = OpSelect %[[#V2F32]] %[[#IsZero]] %[[#]] %[[#NaNSel]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define <2 x float> @minimum_v2f32(<2 x float> %a, <2 x float> %b) {
  %r = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  ret <2 x float> %r
}

; CHECK-LABEL: Begin function maximum_v3f32
; CL: %[[#Max:]] = OpExtInst %[[#V3F32]] %[[#Ext]] fmax
; VK: %[[#Max:]] = OpExtInst %[[#V3F32]] %[[#Ext]] NMax
; CHECK: %[[#NaNSel:]] = OpSelect %[[#V3F32]] %[[#]] %[[#Max]] %[[#]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#]] %[[#NaNSel]] %[[#]]
; CHECK: %[[#Res:]] = OpSelect %[[#V3F32]] %[[#IsZero]] %[[#]] %[[#NaNSel]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define <3 x float> @maximum_v3f32(<3 x float> %a, <3 x float> %b) {
  %r = call <3 x float> @llvm.maximum.v3f32(<3 x float> %a, <3 x float> %b)
  ret <3 x float> %r
}

; CHECK-LABEL: Begin function minimum_v4f32
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#V4F32]]
; CL: %[[#Min:]] = OpExtInst %[[#V4F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; VK: %[[#Min:]] = OpExtInst %[[#V4F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; CL: %[[#Ord:]] = OpOrdered %[[#V4Bool]] %[[#A]] %[[#B]]
; VK: %[[#Ord:]] = OpLogicalNot %[[#V4Bool]]
; CHECK: %[[#NaNSel:]] = OpSelect %[[#V4F32]] %[[#Ord]] %[[#Min]] %[[#]]
; CHECK: %[[#IsZero:]] = OpFOrdEqual %[[#V4Bool]] %[[#NaNSel]] %[[#]]
; CHECK: %[[#Res:]] = OpSelect %[[#V4F32]] %[[#IsZero]] %[[#]] %[[#NaNSel]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define <4 x float> @minimum_v4f32(<4 x float> %a, <4 x float> %b) {
  %r = call <4 x float> @llvm.minimum.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

; CHECK-LABEL: Begin function maximum_v4f32_nnan_nsz
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#V4F32]]
; CHECK-NEXT: OpLabel
; CL-NEXT: %[[#Res:]] = OpExtInst %[[#V4F32]] %[[#Ext]] fmax %[[#A]] %[[#B]]
; VK-NEXT: %[[#Res:]] = OpExtInst %[[#V4F32]] %[[#Ext]] NMax %[[#A]] %[[#B]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define <4 x float> @maximum_v4f32_nnan_nsz(<4 x float> %a, <4 x float> %b) {
  %r = call nnan nsz <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}
