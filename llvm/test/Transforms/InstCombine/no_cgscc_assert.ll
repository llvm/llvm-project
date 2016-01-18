; RUN: opt < %s -inline -instcombine -S | FileCheck %s

; PR21403: http://llvm.org/bugs/show_bug.cgi?id=21403
; When the call to sqrtf is replaced by an intrinsic call to fabs,
; it should not cause a problem in CGSCC. 

define float @bar(float %f) #0 {
  %mul = fmul fast float %f, %f
  %call1 = call float @sqrtf(float %mul) #0
  ret float %call1

; CHECK-LABEL: @bar(
; CHECK-NEXT: call float @llvm.fabs.f32
; CHECK-NEXT: ret float
}

declare float @sqrtf(float) #0

attributes #0 = { readnone "unsafe-fp-math"="true" }
