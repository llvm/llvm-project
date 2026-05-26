; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -enable-loop-vectorization-with-conditions=true \
; RUN:   -conditional-guard-threshold=1 \
; RUN:   -S < %s | FileCheck %s

; Verify that a conditional block with expensive floating-point operations
; (fdiv, sqrt) behind a rarely-taken branch is wrapped in a guard diamond.
; The fdiv and sqrt should only execute when at least one lane is active.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @cond_fdiv_sqrt

; The vector loop backedge comes from the join block.
; CHECK:       vector.body:
; CHECK:         %index = phi i64 [ 0, %vector.ph ], [ %index.next, %if.then.cond.join ]

; Guard: freeze the mask, reduce with OR, branch.
; CHECK:         freeze <4 x i1>
; CHECK:         call i1 @llvm.vector.reduce.or.v4i1(
; CHECK-NEXT:    br i1 {{%.*}}, label %[[GUARDED:.*]], label %if.then.cond.join

; The guarded block contains fdiv and sqrt — only entered when a lane is active.
; CHECK:       [[GUARDED]]:
; CHECK:         fdiv double
; CHECK:         call double @llvm.sqrt.f64(
; CHECK:         br label %if.then.cond.join

; The join block feeds the backedge.
; CHECK:       if.then.cond.join:
; CHECK:         br i1 {{%.*}}, label %middle.block, label %vector.body

define void @cond_fdiv_sqrt(ptr noalias %A, double %threshold, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep = getelementptr inbounds double, ptr %A, i64 %iv
  %val = load double, ptr %gep, align 8
  %cmp = fcmp ogt double %val, %threshold
  br i1 %cmp, label %if.then, label %latch

if.then:
  %div = fdiv double %val, 3.14159265
  %sq  = call double @llvm.sqrt.f64(double %div)
  store double %sq, ptr %gep, align 8
  br label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, %n
  br i1 %exit, label %exit.block, label %loop

exit.block:
  ret void
}

declare double @llvm.sqrt.f64(double)
