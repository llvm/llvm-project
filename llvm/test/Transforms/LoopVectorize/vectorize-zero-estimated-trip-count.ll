; Check that an estimated trip count of zero does not crash or otherwise break
; LoopVectorize behavior while it tries to create runtime memory checks inside
; an outer loop.

; REQUIRES: x86-registered-target
; RUN: opt -passes=loop-vectorize -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Look for basic signs that vectorization ran and produced memory checks.
; CHECK: @test(
; CHECK: vector.memcheck:
; CHECK: vector.body:
; CHECK: inner:

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
