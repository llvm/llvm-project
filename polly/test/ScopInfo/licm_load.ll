; RUN: opt %loadNPMPolly '-passes=polly-custom<prepare;scops>' -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
;    void foo(int n, float A[static const restrict n],
;             float B[static const restrict n], int j) {
;      for (int i = 0; i < n; i++)
;        A[i] = B[j];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B, i32 %j) {
entry:
  %tmp = sext i32 %n to i64
  %cmp1 = icmp slt i64 0, %tmp
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %idxprom = sext i32 %j to i64
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
  %tmp2 = load i32, ptr %arrayidx, align 4
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %indvars.iv2 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.inc ]
  %arrayidx2 = getelementptr inbounds float, ptr %A, i64 %indvars.iv2
  store i32 %tmp2, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv2, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %tmp
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; CHECK:      Invariant Accesses: {
; CHECK-NEXT:   ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [n, j] -> { Stmt_{{[a-zA-Z_]*}}[{{[i0]*}}] -> MemRef_B[j] };
; CHECK-NEXT:       Execution Context: [n, j] -> {  : n > 0 }
; CHECK-NEXT: }
;
; CHECK: Statements {
; CHECK:      Stmt_for_body
; CHECK:     MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:        [n, j] -> { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK:     }
