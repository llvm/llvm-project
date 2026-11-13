; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CL
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CL
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --check-prefixes=CHECK,VK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; llvm.minimum/llvm.maximum propagate NaNs and order -0.0 before +0.0, whereas
; OpenCL.std fmin/fmax and GLSL.std.450 NMin/NMax return the non-NaN operand and
; may return either zero. Check that the missing NaN propagation and signed-zero
; handling are added, and dropped again under nnan and nsz. The expansion matches
; the one in SPIRV-LLVM-Translator. In shaders, copysign is lowered to integer
; operations.

; CL-DAG: %[[#Ext:]] = OpExtInstImport "OpenCL.std"
; VK-DAG: %[[#Ext:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#Bool:]] = OpTypeBool
; CHECK-DAG: %[[#F16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#F64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#V2F32:]] = OpTypeVector %[[#F32]] 2
; CHECK-DAG: %[[#V3F32:]] = OpTypeVector %[[#F32]] 3
; CHECK-DAG: %[[#V4F32:]] = OpTypeVector %[[#F32]] 4
; CHECK-DAG: %[[#V4Bool:]] = OpTypeVector %[[#Bool]] 4
; CHECK-DAG: %[[#NaN16:]] = OpConstant %[[#F16]] 32256
; CHECK-DAG: %[[#NaN32:]] = OpConstant %[[#F32]] 0x1.8p+128
; CHECK-DAG: %[[#NaN64:]] = OpConstant %[[#F64]] 0x1.8p+1024
; CHECK-DAG: %[[#One32:]] = OpConstant %[[#F32]] 1{{$}}
; CL-DAG: %[[#Zero32:]] = OpConstantNull %[[#F32]]
; VK-DAG: %[[#Zero32:]] = OpConstant %[[#F32]] 0{{$}}

; CHECK-LABEL: Begin function minimum_f32
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; CL: %[[#ASign:]] = OpExtInst %[[#F32]] %[[#Ext]] copysign %[[#One32]] %[[#A]]
; VK: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; VK: %[[#ASign:]] = OpBitcast %[[#F32]]
; CHECK: %[[#ANeg:]] = OpFOrdLessThan %[[#Bool]] %[[#ASign]] %[[#Zero32]]
; CHECK: %[[#Sign:]] = OpSelect %[[#F32]] %[[#ANeg]] %[[#A]] %[[#B]]
; CL: %[[#Signed:]] = OpExtInst %[[#F32]] %[[#Ext]] copysign %[[#M]] %[[#Sign]]
; VK: %[[#Signed:]] = OpBitcast %[[#F32]]
; CL: %[[#Uno:]] = OpUnordered %[[#Bool]] %[[#A]] %[[#B]]
; VK: %[[#NaNA:]] = OpIsNan %[[#Bool]] %[[#A]]
; VK: %[[#NaNB:]] = OpIsNan %[[#Bool]] %[[#B]]
; VK: %[[#Uno:]] = OpLogicalOr %[[#Bool]] %[[#NaNA]] %[[#NaNB]]
; CHECK: %[[#Res:]] = OpSelect %[[#F32]] %[[#Uno]] %[[#NaN32]] %[[#Signed]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define float @minimum_f32(float %a, float %b) {
  %r = call float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}

; CHECK-LABEL: Begin function maximum_f32
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] fmax %[[#A]] %[[#B]]
; CL: %[[#ASign:]] = OpExtInst %[[#F32]] %[[#Ext]] copysign %[[#One32]] %[[#A]]
; VK: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] NMax %[[#A]] %[[#B]]
; VK: %[[#ASign:]] = OpBitcast %[[#F32]]
; CHECK: %[[#ANeg:]] = OpFOrdLessThan %[[#Bool]] %[[#ASign]] %[[#Zero32]]
; CHECK: %[[#Sign:]] = OpSelect %[[#F32]] %[[#ANeg]] %[[#B]] %[[#A]]
; CL: %[[#Signed:]] = OpExtInst %[[#F32]] %[[#Ext]] copysign %[[#M]] %[[#Sign]]
; VK: %[[#Signed:]] = OpBitcast %[[#F32]]
; CL: %[[#Uno:]] = OpUnordered %[[#Bool]] %[[#A]] %[[#B]]
; VK: %[[#NaNA:]] = OpIsNan %[[#Bool]] %[[#A]]
; VK: %[[#NaNB:]] = OpIsNan %[[#Bool]] %[[#B]]
; VK: %[[#Uno:]] = OpLogicalOr %[[#Bool]] %[[#NaNA]] %[[#NaNB]]
; CHECK: %[[#Res:]] = OpSelect %[[#F32]] %[[#Uno]] %[[#NaN32]] %[[#Signed]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define float @maximum_f32(float %a, float %b) {
  %r = call float @llvm.maximum.f32(float %a, float %b)
  ret float %r
}

; With nnan, only the signed-zero handling remains.
; CHECK-LABEL: Begin function minimum_f32_nnan
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; VK: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; CHECK: %[[#ANeg:]] = OpFOrdLessThan %[[#Bool]] %[[#]] %[[#Zero32]]
; CHECK: %[[#Sign:]] = OpSelect %[[#F32]] %[[#ANeg]] %[[#A]] %[[#B]]
; CL: %[[#Res:]] = OpExtInst %[[#F32]] %[[#Ext]] copysign %[[#M]] %[[#Sign]]
; VK: %[[#Res:]] = OpBitcast %[[#F32]]
; CHECK-NOT: OpUnordered
; CHECK-NOT: OpIsNan
; CHECK: OpReturnValue %[[#Res]]
define float @minimum_f32_nnan(float %a, float %b) {
  %r = call nnan float @llvm.minimum.f32(float %a, float %b)
  ret float %r
}

; With nsz, only the NaN propagation remains.
; CHECK-LABEL: Begin function minimum_f32_nsz
; CHECK: %[[#A:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#F32]]
; CL: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; CL-NEXT: %[[#Uno:]] = OpUnordered %[[#Bool]] %[[#A]] %[[#B]]
; VK: %[[#M:]] = OpExtInst %[[#F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; VK: %[[#Uno:]] = OpLogicalOr %[[#Bool]]
; CHECK-NEXT: %[[#Res:]] = OpSelect %[[#F32]] %[[#Uno]] %[[#NaN32]] %[[#M]]
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
; CL: %[[#M:]] = OpExtInst %[[#F16]] %[[#Ext]] fmin
; CL: OpExtInst %[[#F16]] %[[#Ext]] copysign
; VK: %[[#M:]] = OpExtInst %[[#F16]] %[[#Ext]] NMin
; CHECK: OpFOrdLessThan
; CHECK: OpSelect %[[#F16]]
; CL: %[[#Signed:]] = OpExtInst %[[#F16]] %[[#Ext]] copysign %[[#M]] %[[#]]
; VK: %[[#Signed:]] = OpBitcast %[[#F16]]
; CHECK: %[[#Res:]] = OpSelect %[[#F16]] %[[#]] %[[#NaN16]] %[[#Signed]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define half @minimum_f16(half %a, half %b) {
  %r = call half @llvm.minimum.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: Begin function maximum_f64
; CL: %[[#M:]] = OpExtInst %[[#F64]] %[[#Ext]] fmax
; CL: OpExtInst %[[#F64]] %[[#Ext]] copysign
; VK: %[[#M:]] = OpExtInst %[[#F64]] %[[#Ext]] NMax
; CHECK: OpFOrdLessThan
; CHECK: OpSelect %[[#F64]]
; CL: %[[#Signed:]] = OpExtInst %[[#F64]] %[[#Ext]] copysign %[[#M]] %[[#]]
; VK: %[[#Signed:]] = OpBitcast %[[#F64]]
; CHECK: %[[#Res:]] = OpSelect %[[#F64]] %[[#]] %[[#NaN64]] %[[#Signed]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define double @maximum_f64(double %a, double %b) {
  %r = call double @llvm.maximum.f64(double %a, double %b)
  ret double %r
}

; CHECK-LABEL: Begin function minimum_v2f32
; CL: %[[#M:]] = OpExtInst %[[#V2F32]] %[[#Ext]] fmin
; CL: OpExtInst %[[#V2F32]] %[[#Ext]] copysign
; VK: %[[#M:]] = OpExtInst %[[#V2F32]] %[[#Ext]] NMin
; CHECK: OpFOrdLessThan
; CHECK: OpSelect %[[#V2F32]]
; CL: %[[#Signed:]] = OpExtInst %[[#V2F32]] %[[#Ext]] copysign %[[#M]] %[[#]]
; VK: %[[#Signed:]] = OpBitcast %[[#V2F32]]
; CHECK: %[[#Res:]] = OpSelect %[[#V2F32]] %[[#]] %[[#]] %[[#Signed]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define <2 x float> @minimum_v2f32(<2 x float> %a, <2 x float> %b) {
  %r = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  ret <2 x float> %r
}

; CHECK-LABEL: Begin function maximum_v3f32
; CL: %[[#M:]] = OpExtInst %[[#V3F32]] %[[#Ext]] fmax
; CL: OpExtInst %[[#V3F32]] %[[#Ext]] copysign
; VK: %[[#M:]] = OpExtInst %[[#V3F32]] %[[#Ext]] NMax
; CHECK: OpFOrdLessThan
; CHECK: OpSelect %[[#V3F32]]
; CL: %[[#Signed:]] = OpExtInst %[[#V3F32]] %[[#Ext]] copysign %[[#M]] %[[#]]
; VK: %[[#Signed:]] = OpBitcast %[[#V3F32]]
; CHECK: %[[#Res:]] = OpSelect %[[#V3F32]] %[[#]] %[[#]] %[[#Signed]]
; CHECK-NEXT: OpReturnValue %[[#Res]]
define <3 x float> @maximum_v3f32(<3 x float> %a, <3 x float> %b) {
  %r = call <3 x float> @llvm.maximum.v3f32(<3 x float> %a, <3 x float> %b)
  ret <3 x float> %r
}

; CHECK-LABEL: Begin function minimum_v4f32
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4F32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#V4F32]]
; CL: %[[#M:]] = OpExtInst %[[#V4F32]] %[[#Ext]] fmin %[[#A]] %[[#B]]
; CL: %[[#ASign:]] = OpExtInst %[[#V4F32]] %[[#Ext]] copysign %[[#]] %[[#A]]
; VK: %[[#M:]] = OpExtInst %[[#V4F32]] %[[#Ext]] NMin %[[#A]] %[[#B]]
; VK: %[[#ASign:]] = OpBitcast %[[#V4F32]]
; CHECK: %[[#ANeg:]] = OpFOrdLessThan %[[#V4Bool]] %[[#ASign]] %[[#]]
; CHECK: %[[#Sign:]] = OpSelect %[[#V4F32]] %[[#ANeg]] %[[#A]] %[[#B]]
; CL: %[[#Signed:]] = OpExtInst %[[#V4F32]] %[[#Ext]] copysign %[[#M]] %[[#Sign]]
; VK: %[[#Signed:]] = OpBitcast %[[#V4F32]]
; CL: %[[#Uno:]] = OpUnordered %[[#V4Bool]] %[[#A]] %[[#B]]
; VK: %[[#NaNA:]] = OpIsNan %[[#V4Bool]] %[[#A]]
; VK: %[[#NaNB:]] = OpIsNan %[[#V4Bool]] %[[#B]]
; VK: %[[#Uno:]] = OpLogicalOr %[[#V4Bool]] %[[#NaNA]] %[[#NaNB]]
; CHECK: %[[#Res:]] = OpSelect %[[#V4F32]] %[[#Uno]] %[[#]] %[[#Signed]]
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
