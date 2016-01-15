; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Check we remove read only statements.
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body_2
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_body_2[i0] : 0 <= i0 <= 99 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_body_2[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body_2[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             { Stmt_for_body_2[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
;
;    int g(int);
;    void f(int *A) {
;      for (int i = 0; i < 100; i++) {
;        (A[i]);
;        /* Split BB */
;        (A[i]);
;        /* Split BB */
;        A[i] += 1;
;        /* Split BB */
;        (A[i]);
;        /* Split BB */
;        (A[i]);
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  br label %for.body.1

for.body.1:
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx2, align 4
  br label %for.body.2

for.body.2:
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx5, align 4
  %add = add nsw i32 %tmp2, 1
  store i32 %add, i32* %arrayidx5, align 4
  br label %for.body.3

for.body.3:
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp3 = load i32, i32* %arrayidx7, align 4
  br label %for.body.4

for.body.4:
  %arrayidx10 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp4 = load i32, i32* %arrayidx10, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
