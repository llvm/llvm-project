
; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=AST
;
;    void f(int *A, int c, int N) {
;      switch (c) {
;      case -1: {
;        for (int j = N; j > 0; j--)
;          A[j] += A[j - 1];
;        break;
;      }
;      case 1: {
;        for (int j = 1; j <= N; j++)
;          A[j] += A[j - 1];
;        break;
;      }
;      }
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [c, N] -> { Stmt_for_body[i0] : c = -1 and 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [c, N] -> { Stmt_for_body[i0] -> [1, i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [c, N] -> { Stmt_for_body[i0] -> MemRef_A[-1 + N - i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [c, N] -> { Stmt_for_body[i0] -> MemRef_A[N - i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [c, N] -> { Stmt_for_body[i0] -> MemRef_A[N - i0] };
; CHECK-NEXT:     Stmt_for_body_7
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [c, N] -> { Stmt_for_body_7[i0] : c = 1 and 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [c, N] -> { Stmt_for_body_7[i0] -> [0, i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [c, N] -> { Stmt_for_body_7[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [c, N] -> { Stmt_for_body_7[i0] -> MemRef_A[1 + i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [c, N] -> { Stmt_for_body_7[i0] -> MemRef_A[1 + i0] };
; CHECK-NEXT: }

; AST:      if (1)
;
; AST:          if (c == 1) {
; AST-NEXT:       for (int c0 = 0; c0 < N; c0 += 1)
; AST-NEXT:         Stmt_for_body_7(c0);
; AST-NEXT:     } else if (c == -1)
; AST-NEXT:       for (int c0 = 0; c0 < N; c0 += 1)
; AST-NEXT:         Stmt_for_body(c0);
;
; AST:      else
; AST-NEXT:     {  /* original code */ }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %c, i32 %N) {
entry:
  br label %entry.split

entry.split:
  switch i32 %c, label %sw.epilog [
    i32 -1, label %sw.bb
    i32 1, label %sw.bb.3
  ]

sw.bb:                                            ; preds = %entry
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %sw.bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ %tmp, %sw.bb ]
  %j.0 = phi i32 [ %N, %sw.bb ], [ %dec, %for.inc ]
  %cmp = icmp sgt i64 %indvars.iv, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = add nsw i32 %j.0, -1
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  %tmp6 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp7 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %tmp7, %tmp6
  store i32 %add, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %dec = add nsw i32 %j.0, -1
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %sw.epilog

sw.bb.3:                                          ; preds = %entry
  %tmp8 = sext i32 %N to i64
  br label %for.cond.5

for.cond.5:                                       ; preds = %for.inc.14, %sw.bb.3
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc.14 ], [ 1, %sw.bb.3 ]
  %cmp6 = icmp sgt i64 %indvars.iv3, %tmp8
  br i1 %cmp6, label %for.end.15, label %for.body.7

for.body.7:                                       ; preds = %for.cond.5
  %tmp9 = add nsw i64 %indvars.iv3, -1
  %arrayidx10 = getelementptr inbounds i32, ptr %A, i64 %tmp9
  %tmp10 = load i32, ptr %arrayidx10, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv3
  %tmp11 = load i32, ptr %arrayidx12, align 4
  %add13 = add nsw i32 %tmp11, %tmp10
  store i32 %add13, ptr %arrayidx12, align 4
  br label %for.inc.14

for.inc.14:                                       ; preds = %for.body.7
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond.5

for.end.15:                                       ; preds = %for.cond.5
  br label %sw.epilog

sw.epilog:                                        ; preds = %for.end.15, %for.end, %entry
  ret void
}
