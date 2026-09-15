; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; llvm.maximumnum differs from llvm.maxnum only in its treatment of signaling
;; NaNs, so it is expanded into a pair of quieting canonicalizations, each of
;; which becomes a multiplication by 1.0, followed by the OpenCL.std fmax that
;; llvm.maxnum already uses.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Float4:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#OneFloat:]] = OpConstant %[[#Float]] 1
; CHECK-DAG: %[[#OneFloat4:]] = OpConstantComposite %[[#Float4]] %[[#OneFloat]] %[[#OneFloat]] %[[#OneFloat]] %[[#OneFloat]]

; CHECK: %[[#X:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Y:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#QX:]] = OpFMul %[[#Float]] %[[#X]] %[[#OneFloat]]
; CHECK: %[[#QY:]] = OpFMul %[[#Float]] %[[#Y]] %[[#OneFloat]]
; CHECK: %[[#Res:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmax %[[#QX]] %[[#QY]]
; CHECK: OpReturnValue %[[#Res]]
define spir_func float @test_maximumnum_float(float %x, float %y) {
  %r = call float @llvm.maximumnum.f32(float %x, float %y)
  ret float %r
}

; CHECK: %[[#Xv:]] = OpFunctionParameter %[[#Float4]]
; CHECK: %[[#Yv:]] = OpFunctionParameter %[[#Float4]]
; CHECK: %[[#QXv:]] = OpFMul %[[#Float4]] %[[#Xv]] %[[#OneFloat4]]
; CHECK: %[[#QYv:]] = OpFMul %[[#Float4]] %[[#Yv]] %[[#OneFloat4]]
; CHECK: %[[#Resv:]] = OpExtInst %[[#Float4]] %[[#ExtInstId]] fmax %[[#QXv]] %[[#QYv]]
; CHECK: OpReturnValue %[[#Resv]]
define spir_func <4 x float> @test_maximumnum_v4float(<4 x float> %x, <4 x float> %y) {
  %r = call <4 x float> @llvm.maximumnum.v4f32(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %r
}

;; With nnan there is no signaling NaN to quiet, so no canonicalization is
;; needed and fmax is emitted on the operands directly.
; CHECK: %[[#Xn:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Yn:]] = OpFunctionParameter %[[#Float]]
; CHECK-NOT: OpFMul
; CHECK: %[[#Resn:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmax %[[#Xn]] %[[#Yn]]
; CHECK: OpReturnValue %[[#Resn]]
define spir_func float @test_maximumnum_nnan(float %x, float %y) {
  %r = call nnan float @llvm.maximumnum.f32(float %x, float %y)
  ret float %r
}

declare float @llvm.maximumnum.f32(float, float)
declare <4 x float> @llvm.maximumnum.v4f32(<4 x float>, <4 x float>)
