; REQUIRES: asserts
; RUN: opt < %s -force-vector-width=2 -force-vector-interleave=1 -loop-vectorize -S --debug-only=loop-vectorize 2>&1 | FileCheck %s

; This test shows extremely high interleaving cost that, probably, should be fixed.
; Due to the high cost, interleaving is not beneficial and the cost model chooses to scalarize
; the load instructions.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

%pair = type { i8, i8 }

; CHECK-LABEL: test
; CHECK: Found an estimated cost of 20 for VF 2 For instruction:   {{.*}} load i8
; CHECK: Found an estimated cost of 0 for VF 2 For instruction:   {{.*}} load i8
; CHECK: vector.body
; CHECK: load i8
; CHECK: br i1 {{.*}}, label %middle.block, label %vector.body

define void @test(%pair* %p, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr %pair, %pair* %p, i64 %i, i32 0
  %tmp1 = load i8, i8* %tmp0, align 1
  %tmp2 = getelementptr %pair, %pair* %p, i64 %i, i32 1
  %tmp3 = load i8, i8* %tmp2, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

