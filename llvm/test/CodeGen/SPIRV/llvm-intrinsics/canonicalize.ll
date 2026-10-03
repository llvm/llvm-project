; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; SPIR-V has no canonicalize instruction. LangRef says llvm.canonicalize
;; "should always be implementable as multiplication by 1.0", so the backend
;; emits an OpFMul against a constant 1.0.

; CHECK-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Float4:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#OneHalf:]] = OpConstant %[[#Half]] 15360
; CHECK-DAG: %[[#OneFloat:]] = OpConstant %[[#Float]] 1
; CHECK-DAG: %[[#OneDouble:]] = OpConstant %[[#Double]] 1
; CHECK-DAG: %[[#OneFloat4:]] = OpConstantComposite %[[#Float4]] %[[#OneFloat]] %[[#OneFloat]] %[[#OneFloat]] %[[#OneFloat]]

; CHECK: %[[#XHalf:]] = OpFunctionParameter %[[#Half]]
; CHECK: %[[#RHalf:]] = OpFMul %[[#Half]] %[[#XHalf]] %[[#OneHalf]]
; CHECK: OpReturnValue %[[#RHalf]]
define spir_func half @test_canonicalize_half(half %x) {
  %r = call half @llvm.canonicalize.f16(half %x)
  ret half %r
}

; CHECK: %[[#XFloat:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#RFloat:]] = OpFMul %[[#Float]] %[[#XFloat]] %[[#OneFloat]]
; CHECK: OpReturnValue %[[#RFloat]]
define spir_func float @test_canonicalize_float(float %x) {
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

; CHECK: %[[#XDouble:]] = OpFunctionParameter %[[#Double]]
; CHECK: %[[#RDouble:]] = OpFMul %[[#Double]] %[[#XDouble]] %[[#OneDouble]]
; CHECK: OpReturnValue %[[#RDouble]]
define spir_func double @test_canonicalize_double(double %x) {
  %r = call double @llvm.canonicalize.f64(double %x)
  ret double %r
}

; CHECK: %[[#XFloat4:]] = OpFunctionParameter %[[#Float4]]
; CHECK: %[[#RFloat4:]] = OpFMul %[[#Float4]] %[[#XFloat4]] %[[#OneFloat4]]
; CHECK: OpReturnValue %[[#RFloat4]]
define spir_func <4 x float> @test_canonicalize_v4float(<4 x float> %x) {
  %r = call <4 x float> @llvm.canonicalize.v4f32(<4 x float> %x)
  ret <4 x float> %r
}

declare half @llvm.canonicalize.f16(half)
declare float @llvm.canonicalize.f32(float)
declare double @llvm.canonicalize.f64(double)
declare <4 x float> @llvm.canonicalize.v4f32(<4 x float>)
