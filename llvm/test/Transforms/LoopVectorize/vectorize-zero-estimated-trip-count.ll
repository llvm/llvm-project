; Check that an estimated trip count of zero does not crash or otherwise break
; LoopVectorize behavior while it tries to create runtime memory checks inside
; an outer loop.

; REQUIRES: x86-registered-target
; RUN: opt -passes=loop-vectorize -S %s | FileCheck %s

; A zero estimated trip count means the outer loop is estimated not to be
; entered.  It must not be used to scale the cost of the memory checks hoisted
; out of it: that would divide by zero.  Check that the fallback trip count of 2
; is used instead.
; RUN: opt -passes=loop-vectorize -disable-output -debug-only=loop-vectorize %s \
; RUN:   2>&1 | FileCheck %s -check-prefix=COST
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Look for basic signs that vectorization ran and produced memory checks.
; CHECK: @test(
; CHECK: vector.memcheck:
; CHECK: vector.body:
; CHECK: inner:

; COST: We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced from 3 to 1

define void @test(ptr addrspace(1) %p, i32 %n) {
entry:
  br label %outer
outer:
  br label %inner
inner:
  %i = phi i32 [ %inc, %inner ], [ 0, %outer ]
  store i32 0, ptr addrspace(1) %p
  %load = load i32, ptr addrspace(1) null
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %inner, label %outer.latch
outer.latch:
  br i1 %cmp, label %outer, label %exit, !llvm.loop !0
exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.estimated_trip_count", i32 0}
