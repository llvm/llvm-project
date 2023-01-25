; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars       -polly-prepare \
; RUN: -polly-invariant-load-hoisting=true -polly-print-scops -disable-output < %s \
; RUN: | FileCheck %s
; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars -licm -polly-prepare \
; RUN: -polly-invariant-load-hoisting=true -polly-print-scops --disable-output< %s \
; RUN: | FileCheck %s
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
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %j to i64
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
  %tmp2 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  store i32 %tmp2, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
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
