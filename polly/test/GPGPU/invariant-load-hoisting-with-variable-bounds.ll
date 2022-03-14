; RUN: opt %loadPolly -polly-invariant-load-hoisting -polly-print-scops -disable-output < %s | FileCheck %s -check-prefix=SCOP


; RUN: opt %loadPolly -S -polly-use-llvm-names -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; SCOP:      Function: f
; SCOP-NEXT: Region: %entry.split---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP-NEXT: Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp1, tmp4] -> { Stmt_entry_split[] -> MemRef_begin[0] };
; SCOP-NEXT:         Execution Context: [tmp1, tmp4] -> {  :  }
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp1, tmp4] -> { Stmt_for_body[i0] -> MemRef_end[0] };
; SCOP-NEXT:         Execution Context: [tmp1, tmp4] -> {  :  }
; SCOP-NEXT: }


; Check that the kernel launch is generated in the host IR.
; This declaration would not have been generated unless a kernel launch exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)

;    void f(int *begin, int *end, int *arr) {
;      for (int i = *begin; i < *end; i++) {
;        arr[i] = 0;
;      }
;    }
;

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

define void @f(i32* %begin, i32* %end, i32* %arr) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp1 = load i32, i32* %begin, align 4
  %tmp41 = load i32, i32* %end, align 4
  %cmp2 = icmp slt i32 %tmp1, %tmp41
  br i1 %cmp2, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.03 = phi i32 [ %tmp1, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %arr, i32 %i.03
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i32 %i.03, 1
  %tmp4 = load i32, i32* %end, align 4
  %cmp = icmp slt i32 %inc, %tmp4
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}
