; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void foo(float *A) {
;      for (long i = 0; i < 16; i++) {
;        A[i] += 1;
;        if (i / 2 == 3)
;          A[i] += 2;
;      }
;    }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body[i0] : 0 <= i0 <= 15 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body[i0] -> [i0, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_if_then
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_if_then[i0] : 6 <= i0 <= 7 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_if_then[i0] -> [i0, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 16
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx0 = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp0 = load float, float* %arrayidx0, align 4
  %add0 = fadd float %tmp0, 2.000000e+00
  store float %add0, float* %arrayidx0, align 4
  %rem1 = sdiv i64 %i.0, 2
  %tobool = icmp ne i64 %rem1, 3
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp = load float, float* %arrayidx, align 4
  %add = fadd float %tmp, 2.000000e+00
  store float %add, float* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
