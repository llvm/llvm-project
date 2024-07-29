; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; Verfy that we do not use the GetElementPtr information to delinearize A
; because of the cast in-between. Use the single-dimensional modeling instead.
;
;    void f(short A[][2]) {
;      for (int i = 0; i < 100; i++)
;        *((long *)&A[4 * i][0]) = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = shl nsw i64 %indvars.iv, 2
  %arrayidx1 = getelementptr inbounds [2 x i16], ptr %A, i64 %tmp, i64 0
  store i64 0, ptr %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}


; CHECK:      Arrays {
; CHECK-NEXT:     i64 MemRef_A[*]; // Element size 8
; CHECK-NEXT: }
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body[i0] -> MemRef_A[2i0] };
; CHECK-NEXT: }
