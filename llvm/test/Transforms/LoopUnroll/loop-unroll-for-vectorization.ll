; Test that the LoopUnrollForVectorization pass correctly unrolls small
; inner loops within a nested loop that carries an explicit vectorization
; hint (llvm.loop.vectorize.enable = true).
;
; RUN: opt < %s -S -passes="loop-unroll-for-vectorization" | FileCheck %s --check-prefix=IR

; IR-LABEL: define void @hot_loop_nest
; The outer loop still has its backedge:
; IR: %cmp1 = icmp ult i64 %i1.next, 3
; IR-NEXT: br i1 %cmp1, label %loop1.header, label %exit
; Inner loop comparisons are gone (fully unrolled):
; IR-NOT: icmp ult i64 %i2.next, 3
; IR-NOT: icmp ult i64 %i3.next, 3
; IR-NOT: icmp ult i64 %i4.next, 3

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Simplified 4-level nested loop (all trip count 3) with a vectorize hint
; on the outermost loop. Models the hot_loop_orig.f90 structure:
;   do loop1 = 1, 3       ! vectorize.enable = true
;     do loop2 = 1, 3
;       do loop3 = 1, 3
;         do loop4 = 1, 3
;           acc += A(loop1,loop3,loop2,loop4) * B(loop3,loop4) * w
;
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
  ; Compute A[i1][i3][i2][i4] -- linearized index
  ; index = i1*27 + i3*9 + i2*3 + i4
  %t1 = mul i64 %i1, 27
  %t2 = mul i64 %i3, 9
  %t3 = mul i64 %i2, 3
  %idx.a = add i64 %t1, %t2
  %idx.a2 = add i64 %idx.a, %t3
  %idx.a3 = add i64 %idx.a2, %i4
  %gep.a = getelementptr double, ptr %A, i64 %idx.a3
  %val.a = load double, ptr %gep.a, align 8
  ; Compute B[i3][i4] -- linearized index = i3*3 + i4
  %b1 = mul i64 %i3, 3
  %idx.b = add i64 %b1, %i4
  %gep.b = getelementptr double, ptr %B, i64 %idx.b
  %val.b = load double, ptr %gep.b, align 8
  ; acc += A[...] * B[...] * w
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

; --- Negative test: loop nest WITHOUT vectorize.enable hint is NOT unrolled ---
; IR-LABEL: define void @no_hint_not_unrolled
; The inner loop backedge should still be present (not unrolled):
; IR: %cmp.inner = icmp ult i64 %j.next, 3
; IR: br i1 %cmp.inner, label %inner.header, label %outer.latch
define void @no_hint_not_unrolled(ptr noalias %A, ptr noalias %out) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc.outer = phi double [ 0.0, %entry ], [ %acc.inner.lcssa, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  %acc.inner = phi double [ %acc.outer, %outer.header ], [ %acc.next, %inner.latch ]
  %idx = add i64 %i, %j
  %gep = getelementptr double, ptr %A, i64 %idx
  %val = load double, ptr %gep, align 8
  %acc.next = fadd fast double %acc.inner, %val
  br label %inner.latch

inner.latch:
  %j.next = add nuw nsw i64 %j, 1
  %cmp.inner = icmp ult i64 %j.next, 3
  br i1 %cmp.inner, label %inner.header, label %outer.latch

outer.latch:
  %acc.inner.lcssa = phi double [ %acc.next, %inner.latch ]
  %i.next = add nuw nsw i64 %i, 1
  %cmp.outer = icmp ult i64 %i.next, 100
  br i1 %cmp.outer, label %outer.header, label %exit

exit:
  store double %acc.inner.lcssa, ptr %out, align 8
  ret void
}

; --- Negative test: inner trip count exceeds limit (17 > default 16) ---
; IR-LABEL: define void @trip_count_too_large
; The inner loop should NOT be unrolled:
; IR: %cmp.big = icmp ult i64 %k.next, 17
; IR: br i1 %cmp.big, label %big.inner.header, label %big.outer.latch
define void @trip_count_too_large(ptr noalias %A, ptr noalias %out) {
entry:
  br label %big.outer.header

big.outer.header:
  %bi = phi i64 [ 0, %entry ], [ %bi.next, %big.outer.latch ]
  %bacc.o = phi double [ 0.0, %entry ], [ %bacc.i.lcssa, %big.outer.latch ]
  br label %big.inner.header

big.inner.header:
  %k = phi i64 [ 0, %big.outer.header ], [ %k.next, %big.inner.latch ]
  %bacc.i = phi double [ %bacc.o, %big.outer.header ], [ %bacc.next, %big.inner.latch ]
  %bidx = add i64 %bi, %k
  %bgep = getelementptr double, ptr %A, i64 %bidx
  %bval = load double, ptr %bgep, align 8
  %bacc.next = fadd fast double %bacc.i, %bval
  br label %big.inner.latch

big.inner.latch:
  %k.next = add nuw nsw i64 %k, 1
  %cmp.big = icmp ult i64 %k.next, 17
  br i1 %cmp.big, label %big.inner.header, label %big.outer.latch

big.outer.latch:
  %bacc.i.lcssa = phi double [ %bacc.next, %big.inner.latch ]
  %bi.next = add nuw nsw i64 %bi, 1
  %cmp.bo = icmp ult i64 %bi.next, 3
  br i1 %cmp.bo, label %big.outer.header, label %big.exit, !llvm.loop !2

big.exit:
  store double %bacc.i.lcssa, ptr %out, align 8
  ret void
}

; Metadata: vectorize.enable=true on the outer loop (loop1)
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
; Metadata for trip_count_too_large outer loop
!2 = distinct !{!2, !1}
