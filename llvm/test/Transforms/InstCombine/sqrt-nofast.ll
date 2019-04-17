; Check that we skip transformations if the attribute unsafe-fp-math
; is not set.
; RUN: opt < %s -instcombine -S | FileCheck %s

define float @mysqrt(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float, float* %x.addr, align 4
  %1 = load float, float* %x.addr, align 4
  %mul = fmul fast float %0, %1
  %2 = call float @llvm.sqrt.f32(float %mul)
  ret float %2
}

declare float @llvm.sqrt.f32(float) #1

; CHECK: define float @mysqrt(float %x, float %y) {
; CHECK: entry:
; CHECK:   %mul = fmul fast float %x, %x
; CHECK:   %0 = call float @llvm.sqrt.f32(float %mul)
; CHECK:   ret float %0
; CHECK: }
