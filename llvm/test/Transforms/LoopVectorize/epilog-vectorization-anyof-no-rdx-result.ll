; RUN: opt -passes=loop-vectorize -S %s | FileCheck %s
;
; Verify that epilogue vectorization does not crash when a VPReductionPHIRecipe
; (AnyOf reduction) has no ComputeReductionResult because the exit value was
; simplified to a constant.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @main
; CHECK: vec.epilog.vector.body:
; CHECK: middle.block:
define i8 @main() {
entry:
  br label %loop

exit:
  ret i8 %sel

loop:
  %phi.rdx = phi i8 [ %sel, %loop ], [ 0, %entry ]
  %iv = phi i32 [ %iv.next, %loop ], [ 1, %entry ]
  %cmp = icmp sgt i32 0, 0
  %sel = select i1 %cmp, i8 0, i8 %phi.rdx
  %iv.next = add i32 %iv, 1
  %exit.cond = icmp eq i32 %iv.next, 0
  br i1 %exit.cond, label %exit, label %loop

; uselistorder directives
  uselistorder i8 %sel, { 1, 0 }
}
