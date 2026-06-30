; OptimizeMaskedMemory on AArch64 NEON: a predicated store with no guaranteed
; access at the same SCEV must NOT be rewritten -- the unconditional
; load-blend-store would introduce a fault on lanes the original code never
; touched.
;
; RUN: opt < %s -passes=loop-vectorize -mtriple=aarch64-unknown-linux-gnu \
; RUN:     -mattr=+neon,-sve -enable-masked-memory-optimization -S \
; RUN: | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @neon_not_safe
; CHECK-NOT: blend.load
; CHECK-NOT: blend.sel
define void @neon_not_safe(ptr noalias %A, ptr noalias %cond) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep_cond = getelementptr inbounds i32, ptr %cond, i64 %iv
  %cv = load i32, ptr %gep_cond, align 4
  %cmp = icmp ne i32 %cv, 0
  br i1 %cmp, label %if.then, label %loop.latch

if.then:
  ; Predicated store with no companion guaranteed access at the same SCEV
  ; anywhere else in the loop body.
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
