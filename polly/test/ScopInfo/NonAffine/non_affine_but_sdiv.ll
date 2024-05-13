; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_A[o0] : -4 + N + 5i0 <= 5o0 <= N + 5i0 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_A[o0] : -N + 5i0 <= 5o0 <= 4 - N + 5i0 };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++)
;        A[i] = A[i + (N / 5)] + A[i + (N / -5)];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %div = sdiv i32 %N, 5
  %tmp1 = trunc i64 %indvars.iv to i32
  %add = add nsw i32 %tmp1, %div
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  %tmp2 = load i32, ptr %arrayidx, align 4
  %div1 = sdiv i32 %N, -5
  %tmp3 = trunc i64 %indvars.iv to i32
  %add2 = add nsw i32 %tmp3, %div1
  %idxprom3 = sext i32 %add2 to i64
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %idxprom3
  %tmp4 = load i32, ptr %arrayidx4, align 4
  %add5 = add nsw i32 %tmp2, %tmp4
  %arrayidx7 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add5, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
