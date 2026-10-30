; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=FPFastMathMode
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define spir_func float @Test(float %x, float %y) {
entry:
  %0 = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %0
}

; nsz and reassoc would let fmax change the sign of two same-sign zeros.
define spir_func float @TestFMF(float %x, float %y) {
entry:
  %0 = call nnan nsz float @llvm.maxnum.f32(float %x, float %y)
  %1 = call reassoc float @llvm.maxnum.f32(float %0, float %y)
  ret float %1
}

; CHECK: OpDecorate %[[#nnan:]] FPFastMathMode NotNaN{{$}}
; CHECK: OpFunction
; CHECK: %[[#x:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#y:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#res:]] = OpExtInst %[[#]] %[[#]] fmax %[[#x]] %[[#y]]
; CHECK: OpReturnValue %[[#res]]
; CHECK: %[[#nnan]] = OpExtInst %[[#]] %[[#]] fmax

declare float @llvm.maxnum.f32(float, float)
