; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; llvm.minimumnum differs from llvm.minnum only in its treatment of signaling
;; NaNs, so it is expanded into a pair of quieting canonicalizations, each of
;; which becomes a multiplication by 1.0, followed by the OpenCL.std fmin that
;; llvm.minnum already uses.

; CHECK-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#OneFloat:]] = OpConstant %[[#Float]] 1
; CHECK-DAG: %[[#OneDouble:]] = OpConstant %[[#Double]] 1

; CHECK: %[[#X:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Y:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#QX:]] = OpFMul %[[#Float]] %[[#X]] %[[#OneFloat]]
; CHECK: %[[#QY:]] = OpFMul %[[#Float]] %[[#Y]] %[[#OneFloat]]
; CHECK: %[[#Res:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmin %[[#QX]] %[[#QY]]
; CHECK: OpReturnValue %[[#Res]]
define spir_func float @test_minimumnum_float(float %x, float %y) {
  %r = call float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

; CHECK: %[[#Xd:]] = OpFunctionParameter %[[#Double]]
; CHECK: %[[#Yd:]] = OpFunctionParameter %[[#Double]]
; CHECK: %[[#QXd:]] = OpFMul %[[#Double]] %[[#Xd]] %[[#OneDouble]]
; CHECK: %[[#QYd:]] = OpFMul %[[#Double]] %[[#Yd]] %[[#OneDouble]]
; CHECK: %[[#Resd:]] = OpExtInst %[[#Double]] %[[#ExtInstId]] fmin %[[#QXd]] %[[#QYd]]
; CHECK: OpReturnValue %[[#Resd]]
define spir_func double @test_minimumnum_double(double %x, double %y) {
  %r = call double @llvm.minimumnum.f64(double %x, double %y)
  ret double %r
}

declare float @llvm.minimumnum.f32(float, float)
declare double @llvm.minimumnum.f64(double, double)
