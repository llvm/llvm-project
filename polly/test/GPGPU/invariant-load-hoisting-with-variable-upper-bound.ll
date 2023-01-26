; RUN: opt %loadPolly -polly-invariant-load-hoisting -polly-print-scops -disable-output < %s | FileCheck %s -check-prefix=SCOP
; RUN: opt %loadPolly -S -polly-use-llvm-names -polly-codegen-ppcg -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; Check that we detect a scop with invariant accesses.
; SCOP:      Function: f
; SCOP-NEXT: Region: %entry.split---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP-NEXT: Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp2] -> { Stmt_for_body[i0] -> MemRef_idx[0] };
; SCOP-NEXT:         Execution Context: [tmp2] -> {  :  }
; SCOP-NEXT: }

; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(ptr, i32, i32, i32, i32, i32, ptr)

; Check if we generate GPU code for simple loop with variable upper bound.
; This always worked, but have this test to prevent regressions.
;    void f(int *idx, int *arr) {
;      for (int i = 0; i < *idx; i++) {
;        arr[i] = 0;
;      }
;    }
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %idx, ptr %arr) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp21 = load i32, ptr %idx, align 4
  %cmp2 = icmp sgt i32 %tmp21, 0
  br i1 %cmp2, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %arr, i64 %indvars.iv
  store i32 0, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp2 = load i32, ptr %idx, align 4
  %0 = sext i32 %tmp2 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %0
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}
