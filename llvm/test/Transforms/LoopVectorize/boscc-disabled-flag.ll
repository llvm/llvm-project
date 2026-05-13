; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -enable-loop-vectorization-with-conditions=false \
; RUN:   -S < %s | FileCheck %s

; Verify that the conditional block guard transformation is NOT applied
; when disabled via -enable-loop-vectorization-with-conditions=false.
; No any-of guard reduction or cond.join block should appear.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @cond_store_no_guard

; The loop should be vectorized but without any guard diamond.
; CHECK-NOT:     call i1 @llvm.vector.reduce.or.v4i1(
; CHECK-NOT:     if.then.cond.join

; The vector loop backedge goes directly to vector.body (no join block).
; CHECK:       vector.body:
; CHECK:         store i32

define void @cond_store_no_guard(ptr noalias %A, ptr noalias %B, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep.a = getelementptr inbounds i32, ptr %A, i64 %iv
  %val = load i32, ptr %gep.a, align 4
  %cmp = icmp sgt i32 %val, 100
  br i1 %cmp, label %if.then, label %latch

if.then:
  %add = add nsw i32 %val, 42
  %mul = mul nsw i32 %add, 3
  %gep.b = getelementptr inbounds i32, ptr %B, i64 %iv
  store i32 %mul, ptr %gep.b, align 4
  br label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, %n
  br i1 %exit, label %exit.block, label %loop

exit.block:
  ret void
}
