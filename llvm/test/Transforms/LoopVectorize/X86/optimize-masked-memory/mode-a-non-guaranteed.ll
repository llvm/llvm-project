; Phase-1 / Mode-A: a predicated store with no guaranteed access at the same
; SCEV must NOT be rewritten -- the load-blend-store would introduce a fault.
;
; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu \
; RUN:     -mcpu=znver3 -enable-masked-memory-optimization -S \
; RUN: | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @phase1_not_safe
; CHECK: call void @llvm.masked.store
; CHECK-NOT: blend.load
; CHECK-NOT: blend.sel
define void @phase1_not_safe(ptr noalias %A, ptr noalias %cond) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep_cond = getelementptr inbounds i32, ptr %cond, i64 %iv
  %cv = load i32, ptr %gep_cond, align 4
  %cmp = icmp ne i32 %cv, 0
  br i1 %cmp, label %if.then, label %loop.latch

if.then:
  ; Predicated store with no companion guaranteed access at the same
  ; SCEV anywhere else in the loop body.
  %gep_A = getelementptr inbounds i32, ptr %A, i64 %iv
  %val = trunc i64 %iv to i32
  store i32 %val, ptr %gep_A, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
