; RUN: opt %loadPolly -polly-invariant-load-hoisting -polly-print-scops -disable-output < %s | FileCheck %s -check-prefix=SCOP

; RUN: opt %loadPolly -S -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=HOST-IR


; REQUIRES: pollyacc

; Check that we detect a scop.
; SCOP:      Function: f
; SCOP-NEXT: Region: %for.body---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP-NEXT: Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp] -> { Stmt_for_body[i0] -> MemRef_control[0] };
; SCOP-NEXT:         Execution Context: [tmp] -> {  :  }
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp] -> { Stmt_if_then[i0] -> MemRef_readarr[0] };
; SCOP-NEXT:         Execution Context: [tmp] -> {  : tmp >= 4 }
; SCOP-NEXT: }

; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)

; This test makes sure that such an access pattern is handled correctly
; by PPCGCodeGeneration. It appears that not calling `preloadInvariantLoads`
; was the main reason that caused this test case to crash.
;
; void f(int *arr, const int *control, const int *readarr) {
;     for(int i = 0; i < 1000; i++) {
;         int t = 0;
;         if (*control > 3) {
;             t += *readarr;
;         }
;         arr[i] = t;
;     }
; }


target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.12.0"
define void @f(i32* %arr, i32* %control, i32* %readarr) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %if.end
  %i.01 = phi i32 [ 0, %entry.split ], [ %inc, %if.end ]
  %tmp = load i32, i32* %control, align 4
  %cmp1 = icmp sgt i32 %tmp, 3
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %tmp1 = load i32, i32* %readarr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %t.0 = phi i32 [ %tmp1, %if.then ], [ 0, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %arr, i32 %i.01
  store i32 %t.0, i32* %arrayidx, align 4
  %inc = add nuw nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %if.end
  ret void
}
