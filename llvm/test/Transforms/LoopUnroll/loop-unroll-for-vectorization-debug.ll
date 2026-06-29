; Test debug output of the LoopUnrollForVectorization pass.
;
; RUN: opt < %s -S -passes="loop-unroll-for-vectorization" -debug-only=loop-unroll-for-vectorization 2>&1 | FileCheck %s
;
; REQUIRES: asserts

; CHECK: LoopUnrollForVec: Found {{[0-9]+}} outer loop(s) with unrollable inner loops in hot_loop_nest
; CHECK: LoopUnrollForVec: {{[0-9]+}} inner loop candidate(s) in outer loop at loop1.header
; CHECK: LoopUnrollForVec: Unrolling loop at loop4.header (trip count 3)
; CHECK: LoopUnrollForVec: Unrolling loop at loop3.header (trip count 3)
; CHECK: LoopUnrollForVec: Unrolling loop at loop2.header (trip count 3)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hot_loop_nest(ptr noalias %A, ptr noalias %B, double %w, ptr noalias %out) {
entry:
  br label %loop1.header

loop1.header:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %acc1 = phi double [ 0.0, %entry ], [ %acc1.next, %loop1.latch ]
  br label %loop2.header

loop2.header:
  %i2 = phi i64 [ 0, %loop1.header ], [ %i2.next, %loop2.latch ]
  %acc2 = phi double [ %acc1, %loop1.header ], [ %acc2.next, %loop2.latch ]
  br label %loop3.header

loop3.header:
  %i3 = phi i64 [ 0, %loop2.header ], [ %i3.next, %loop3.latch ]
  %acc3 = phi double [ %acc2, %loop2.header ], [ %acc3.next, %loop3.latch ]
  br label %loop4.header

loop4.header:
  %i4 = phi i64 [ 0, %loop3.header ], [ %i4.next, %loop4.latch ]
  %acc4 = phi double [ %acc3, %loop3.header ], [ %acc4.next, %loop4.latch ]
  %t1 = mul i64 %i1, 27
  %t2 = mul i64 %i3, 9
  %t3 = mul i64 %i2, 3
  %idx.a = add i64 %t1, %t2
  %idx.a2 = add i64 %idx.a, %t3
  %idx.a3 = add i64 %idx.a2, %i4
  %gep.a = getelementptr double, ptr %A, i64 %idx.a3
  %val.a = load double, ptr %gep.a, align 8
  %b1 = mul i64 %i3, 3
  %idx.b = add i64 %b1, %i4
  %gep.b = getelementptr double, ptr %B, i64 %idx.b
  %val.b = load double, ptr %gep.b, align 8
  %mul1 = fmul fast double %val.a, %val.b
  %mul2 = fmul fast double %mul1, %w
  %acc4.next = fadd fast double %acc4, %mul2
  br label %loop4.latch

loop4.latch:
  %i4.next = add nuw nsw i64 %i4, 1
  %cmp4 = icmp ult i64 %i4.next, 3
  br i1 %cmp4, label %loop4.header, label %loop3.latch

loop3.latch:
  %acc3.next = phi double [ %acc4.next, %loop4.latch ]
  %i3.next = add nuw nsw i64 %i3, 1
  %cmp3 = icmp ult i64 %i3.next, 3
  br i1 %cmp3, label %loop3.header, label %loop2.latch

loop2.latch:
  %acc2.next = phi double [ %acc3.next, %loop3.latch ]
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, 3
  br i1 %cmp2, label %loop2.header, label %loop1.latch

loop1.latch:
  %acc1.next = phi double [ %acc2.next, %loop2.latch ]
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, 3
  br i1 %cmp1, label %loop1.header, label %exit, !llvm.loop !0

exit:
  store double %acc1.next, ptr %out, align 8
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
