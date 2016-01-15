; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-ast -analyze < %s | FileCheck %s --check-prefix=AST
;
;    void f(int *A, int *B, int N) {
;      for (int i = 0; i < N; i++) {
;        A[i]++;
;        switch (N) {
;        case 0:
;          B[i]++;
;          break;
;        default:
;          return;
;        }
;      }
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_body[0] : N >= 1 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> [0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_body[i0] -> MemRef_A[0] };
; CHECK-NEXT: }

; AST:      if (1)
;
; AST:          if (N >= 1)
; AST-NEXT:       Stmt_for_body(0);
;
; AST:      else
; AST-NEXT:     {  /* original code */ }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32* %B, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, i32* %arrayidx, align 4
  switch i32 %N, label %sw.default [
    i32 0, label %sw.bb
  ]

sw.bb:                                            ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx2, align 4
  %inc3 = add nsw i32 %tmp2, 1
  store i32 %inc3, i32* %arrayidx2, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %for.body
  br label %for.end

sw.epilog:                                        ; preds = %sw.bb
  br label %for.inc

for.inc:                                          ; preds = %sw.epilog
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end.loopexit:                                 ; preds = %for.cond
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %sw.default
  ret void
}
