; RUN: opt %loadPolly -polly-scops -polly-print-simplify -disable-output < %s | FileCheck %s
;
; The use of %sum.next by %phi counts as an escaping use.
; Don't remove the scalar write of %sum.next.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float* %A) {
entry:
  br label %header

header:
  fence seq_cst
  br i1 true, label %body, label %exit

body:
  %i = phi i64 [ 0, %header ], [ %next, %body ]
  %sum = phi float [ 0.0, %header ], [ %sum.next, %body ]
  %arrayidx = getelementptr float, float* %A, i64 %i
  %next = add nuw nsw i64 %i, 1
  %val = load float, float* %arrayidx
  %sum.next = fadd float %sum, %val
  %cond = icmp ne i64 %i, 100
  br i1 %cond, label %body, label %after

after:
  br label %exit

exit:
  %phi = phi float [%sum.next, %after], [0.0, %header]
  ret float %phi
}


; CHECK: Statistics {
; CHECK:     Dead accesses removed: 0
; CHECK: }
