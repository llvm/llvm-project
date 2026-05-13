; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -enable-loop-vectorization-with-conditions=true \
; RUN:   -conditional-guard-threshold=1 \
; RUN:   -S < %s | FileCheck %s

; Verify that a conditional store behind a branch in the scalar loop gets
; wrapped in a guard diamond after vectorization.  The guard reduces the
; vector mask with any-of (vector.reduce.or) and branches around the
; predicated block when no lane is active.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @cond_store_i32

; The vector loop backedge comes from the join block.
; CHECK:       vector.body:
; CHECK:         %index = phi i64 [ 0, %vector.ph ], [ %index.next, %if.then.cond.join ]

; Guard: freeze the masks, OR them, horizontal reduce, branch.
; CHECK:         freeze <4 x i1>
; CHECK:         call i1 @llvm.vector.reduce.or.v4i1(
; CHECK-NEXT:    br i1 {{%.*}}, label %[[GUARDED:.*]], label %if.then.cond.join

; The guarded block contains the predicated stores.
; CHECK:       [[GUARDED]]:
; CHECK:         store i32
; CHECK:         br label %if.then.cond.join

; The join block merges control flow and feeds the backedge.
; CHECK:       if.then.cond.join:
; CHECK:         br i1 {{%.*}}, label %middle.block, label %vector.body

define void @cond_store_i32(ptr noalias %A, ptr noalias %B, i64 %n) {
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
