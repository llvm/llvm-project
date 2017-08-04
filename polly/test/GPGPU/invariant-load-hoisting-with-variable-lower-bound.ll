; RUN: opt %loadPolly -analyze -polly-use-llvm-names -polly-scops \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=SCOP


; RUN: opt %loadPolly -S -polly-use-llvm-names -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; Check that we detect a scop with invariant accesses.
; SCOP:      Function: f
; SCOP-NEXT: Region: %entry.split---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP-NEXT: Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [beginval] -> { Stmt_entry_split[] -> MemRef_begin[0] };
; SCOP-NEXT:         Execution Context: [beginval] -> {  :  }
; SCOP-NEXT: }

; Check that the kernel launch is generated in the host IR.
; This declaration would not have been generated unless a kernel launch exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)

; 
; void f(int *begin, int *arr) {
;     for (int i = *begin; i < 100; i++) {
;         arr[i] = 0;
;     }
; }

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

define void @f(i32* %begin, i32* %arr) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %beginval = load i32, i32* %begin, align 4
  %cmp1 = icmp slt i32 %beginval, 100
  br i1 %cmp1, label %for.body, label %for.end



for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %ival = phi i32 [ %beginval, %entry.split ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %arr, i32 %ival
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i32 %ival, 1
  %cmp = icmp slt i32 %ival, 99
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}
