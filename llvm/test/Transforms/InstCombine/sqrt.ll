; RUN: opt -S -instcombine < %s | FileCheck %s

define float @test1(float %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: @test1(
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

; PR8096
define float @test2(float %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: @test2(
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

; rdar://9763193
; Can't fold (fptrunc (sqrt (fpext x))) -> (sqrtf x) since there is another
; use of sqrt result.
define float @test3(float* %v) nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test3(
; CHECK: sqrt(
; CHECK-NOT: sqrtf(
; CHECK: fptrunc
  %arrayidx13 = getelementptr inbounds float, float* %v, i64 2
  %tmp14 = load float, float* %arrayidx13
  %mul18 = fmul float %tmp14, %tmp14
  %add19 = fadd float undef, %mul18
  %conv = fpext float %add19 to double
  %call34 = call double @sqrt(double %conv) readnone
  %call36 = call i32 (double) @foo(double %call34) nounwind
  %conv38 = fptrunc double %call34 to float
  ret float %conv38
}

declare i32 @foo(double)

declare double @sqrt(double) readnone
