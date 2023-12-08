; RUN: opt < %s -loop-reduce -S -mtriple=x86_64-unknown-unknown | FileCheck %s
;
; Test LSR's OptimizeShadowIV. Handle a floating-point IV with a
; nonzero initial value.
; rdar://9786536

; Provide legal integer types.
target datalayout = "n8:16:32:64"


; First, make sure LSR doesn't crash on an empty IVUsers list.
; CHECK-LABEL: @dummyIV(
; CHECK-NOT: phi
; CHECK-NOT: sitofp
; CHECK: br
define void @dummyIV() nounwind {
entry:
  br label %loop

loop:
  %i.01 = phi i32 [ -39, %entry ], [ %inc, %loop ]
  %conv = sitofp i32 %i.01 to double
  %inc = add nsw i32 %i.01, 1
  br i1 undef, label %loop, label %for.end

for.end:
  unreachable
}

; Now check that the computed double constant is correct.
; CHECK-LABEL: @doubleIV(
; CHECK: phi double [ -3.900000e+01, %entry ]
; CHECK: br
define void @doubleIV() nounwind {
entry:
  br label %loop

loop:
  %i.01 = phi i32 [ -39, %entry ], [ %inc, %loop ]
  %conv = sitofp i32 %i.01 to double
  %div = fdiv double %conv, 4.000000e+01
  %inc = add nsw i32 %i.01, 1
  br i1 undef, label %loop, label %for.end

for.end:
  unreachable
}
