; RUN: opt < %s -inline -instcombine -S | FileCheck %s

; PR21403: http://llvm.org/bugs/show_bug.cgi?id=21403
; When the call to sqrtf is replaced by an intrinsic call to fabs,
; it should not cause a problem in CGSCC. 

define float @bar(float %f) #0 {
  %mul = fmul fast float %f, %f
  %call1 = call fast float @sqrtf(float %mul)
  ret float %call1

; CHECK-LABEL: @bar(
; CHECK-NEXT: call fast float @llvm.fabs.f32
; CHECK-NEXT: ret float
}

declare float @sqrtf(float)

