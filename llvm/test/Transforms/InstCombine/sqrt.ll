; RUN: opt -S -instcombine %s | FileCheck %s

define float @test1(float %x) nounwind readnone ssp {
entry:
; CHECK: @test1
; CHECK-NOT: fpext
; CHECK-NOT: sqrt(
; CHECK: sqrtf(
; CHECK-NOT: fptrunc
  %conv = fpext float %x to double                ; <double> [#uses=1]
  %call = tail call double @sqrt(double %conv) readnone nounwind ; <double> [#uses=1]
  %conv1 = fptrunc double %call to float          ; <float> [#uses=1]
; CHECK: ret float
  ret float %conv1
}

declare double @sqrt(double)

; PR8096
define float @test2(float %x) nounwind readnone ssp {
entry:
; CHECK: @test2
; CHECK-NOT: fpext
; CHECK-NOT: sqrt(
; CHECK: sqrtf(
; CHECK-NOT: fptrunc
  %conv = fpext float %x to double                ; <double> [#uses=1]
  %call = tail call double @sqrt(double %conv) nounwind ; <double> [#uses=1]
  %conv1 = fptrunc double %call to float          ; <float> [#uses=1]
; CHECK: ret float
  ret float %conv1
}
