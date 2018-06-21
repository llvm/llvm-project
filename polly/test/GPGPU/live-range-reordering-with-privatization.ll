  ; RUN: opt %loadPolly -polly-use-llvm-names -polly-scops \
; RUN: -polly-invariant-load-hoisting -polly-codegen-ppcg \
; RUN: -polly-acc-dump-code -disable-output \
; RUN:   < %s | FileCheck %s -check-prefix=CODE

; RUN: opt %loadPolly -polly-use-llvm-names -polly-scops \
; RUN: -polly-invariant-load-hoisting -polly-codegen-ppcg \
; RUN: -polly-acc-dump-kernel-ir -disable-output \
; RUN:   < %s | FileCheck %s -check-prefix=KERNELIR

; REQUIRES: pollyacc

;    void f(const int *end, int *arr, const int *control, const int *readarr) {
;      for (int i = 0; i < *end; i++) {
;        int t = 0;
;        if (*control > 3) {
;          t += readarr[i];
;        }
;        arr[i] = t;
;      }
;    }

; This test case tests the ability to infer that `t` is local to each loop
; iteration, and can therefore be privatized.

; CODE: # kernel0
; CODE-NEXT: for (int c0 = 0; c0 <= (tmp - 32 * b0 - 1) / 1048576; c0 += 1)
; CODE-NEXT:   if (tmp >= 32 * b0 + t0 + 1048576 * c0 + 1) {
; CODE-NEXT:     Stmt_for_body_last(32 * b0 + t0 + 1048576 * c0);
; CODE-NEXT:     if (tmp1 >= 4)
; CODE-NEXT:       Stmt_if_then(32 * b0 + t0 + 1048576 * c0);
; CODE-NEXT:     Stmt_if_end(32 * b0 + t0 + 1048576 * c0);
; CODE-NEXT:   }

; KERNELIR: %private_array = alloca i32

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.12.0"

define void @f(i32* %end, i32* %arr, i32* %control, i32* %readarr) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp3 = load i32, i32* %end, align 4
  %cmp4 = icmp sgt i32 %tmp3, 0
  br i1 %cmp4, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %if.end
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %if.end ]
  %tmp1 = load i32, i32* %control, align 4
  %cmp1 = icmp sgt i32 %tmp1, 3
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %readarr, i32 %i.05
  %tmp2 = load i32, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %t.0 = phi i32 [ %tmp2, %if.then ], [ 0, %for.body ]
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i32 %i.05
  store i32 %t.0, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %tmp = load i32, i32* %end, align 4
  %cmp = icmp slt i32 %inc, %tmp
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %if.end
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}

