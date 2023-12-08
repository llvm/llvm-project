; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=AST
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        switch (i) {
;        case 0:
;          A[i] += 1;
;          break;
;        case 1:
;          A[i] += 2;
;          break;
;        case 2:
;          A[i] += 3;
;          break;
;        case 3:
;          A[i] += 4;
;          break;
;        default:;
;        }
;      }
;    }

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_sw_bb
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb[0] : N > 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb[i0] -> [0, 3] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb[i0] -> MemRef_A[0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb[i0] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_sw_bb_1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_1[1] : N >= 2 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_1[i0] -> [1, 2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_1[i0] -> MemRef_A[1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_1[i0] -> MemRef_A[1] };
; CHECK-NEXT:     Stmt_sw_bb_5
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_5[2] : N >= 3 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_5[i0] -> [2, 1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_5[i0] -> MemRef_A[2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_5[i0] -> MemRef_A[2] };
; CHECK-NEXT:     Stmt_sw_bb_9
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_9[3] : N >= 4 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_9[i0] -> [3, 0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_9[i0] -> MemRef_A[3] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_sw_bb_9[i0] -> MemRef_A[3] };
; CHECK-NEXT: }

; AST:      if (1)
;
; AST:          if (N >= 1) {
; AST-NEXT:       Stmt_sw_bb(0);
; AST-NEXT:       if (N >= 2) {
; AST-NEXT:         Stmt_sw_bb_1(1);
; AST-NEXT:         if (N >= 3) {
; AST-NEXT:           Stmt_sw_bb_5(2);
; AST-NEXT:           if (N >= 4)
; AST-NEXT:             Stmt_sw_bb_9(3);
; AST-NEXT:         }
; AST-NEXT:       }
; AST-NEXT:     }
;
; AST:      else
; AST-NEXT:     {  /* original code */ }

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
  %tmp1 = trunc i64 %indvars.iv to i32
  switch i32 %tmp1, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb.1
    i32 2, label %sw.bb.5
    i32 3, label %sw.bb.9
  ]

sw.bb:                                            ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp2, 1
  store i32 %add, ptr %arrayidx, align 4
  br label %sw.epilog

sw.bb.1:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp3 = load i32, ptr %arrayidx3, align 4
  %add4 = add nsw i32 %tmp3, 2
  store i32 %add4, ptr %arrayidx3, align 4
  br label %sw.epilog

sw.bb.5:                                          ; preds = %for.body
  %arrayidx7 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp4 = load i32, ptr %arrayidx7, align 4
  %add8 = add nsw i32 %tmp4, 3
  store i32 %add8, ptr %arrayidx7, align 4
  br label %sw.epilog

sw.bb.9:                                          ; preds = %for.body
  %arrayidx11 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp5 = load i32, ptr %arrayidx11, align 4
  %add12 = add nsw i32 %tmp5, 4
  store i32 %add12, ptr %arrayidx11, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %for.body
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb.9, %sw.bb.5, %sw.bb.1, %sw.bb
  br label %for.inc

for.inc:                                          ; preds = %sw.epilog
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
