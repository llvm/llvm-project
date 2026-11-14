; The nest-size cap (-unroll-max-nest-loops) suppresses heuristic peeling and
; partial unrolling of loops in an over-cap nest, not just runtime unrolling.
; Each flavor is checked with the cap disabled (0, transform happens) and with
; the cap at 1, where the two-loop nest is over the limit and the transform is
; suppressed. Forced transforms (e.g. -unroll-force-peel-count, -unroll-count)
; are intentionally not gated, mirroring the pragma-override behavior.
;
; RUN: opt < %s -S -passes=loop-unroll -unroll-allow-peeling=true \
; RUN:   -unroll-allow-loop-nests-peeling=true -unroll-max-nest-loops=0 \
; RUN:   | FileCheck %s -check-prefix=PEEL
; RUN: opt < %s -S -passes=loop-unroll -unroll-allow-peeling=true \
; RUN:   -unroll-allow-loop-nests-peeling=true -unroll-max-nest-loops=1 \
; RUN:   | FileCheck %s -check-prefix=PEELCAP
; RUN: opt < %s -S -passes=loop-unroll -unroll-allow-partial \
; RUN:   -unroll-max-nest-loops=0 | FileCheck %s -check-prefix=PARTIAL
; RUN: opt < %s -S -passes=loop-unroll -unroll-allow-partial \
; RUN:   -unroll-max-nest-loops=1 | FileCheck %s -check-prefix=PARTIALCAP

target datalayout = "e-p:64:64:64-i64:64:64-n8:16:32:64-S128"

; The inner loop's %p becomes invariant after the first iteration, making it a
; peeling candidate. Peeling fires only when the nest is within the cap.
; PEEL-LABEL:    @nest_peel(
; PEEL:          inner.peel
; PEELCAP-LABEL: @nest_peel(
; PEELCAP-NOT:   inner.peel

define void @nest_peel(ptr %a, i64 %n, i64 %m) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %p = phi i32 [ 0, %outer ], [ 1, %inner ]
  %idx = getelementptr inbounds i32, ptr %a, i64 %j
  %v = load i32, ptr %idx
  %v2 = add i32 %v, %p
  store i32 %v2, ptr %idx
  %j.next = add i64 %j, 1
  %inner.cond = icmp ult i64 %j.next, %m
  br i1 %inner.cond, label %inner, label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %outer.cond = icmp ult i64 %i.next, %n
  br i1 %outer.cond, label %outer, label %exit, !llvm.loop !0

exit:
  ret void
}

; The inner loop has a compile-time trip count (100). With partial unrolling
; allowed it is unrolled by a factor, producing duplicated body values such as
; %j.next.1. The cap suppresses this for an over-cap nest.
; PARTIAL-LABEL:    @nest_partial(
; PARTIAL:          %j.next.1
; PARTIALCAP-LABEL: @nest_partial(
; PARTIALCAP-NOT:   %j.next.1

define void @nest_partial(ptr %a, i64 %n) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx = getelementptr inbounds i32, ptr %a, i64 %j
  %v = load i32, ptr %idx
  %v2 = add i32 %v, 1
  store i32 %v2, ptr %idx
  %j.next = add i64 %j, 1
  %inner.cond = icmp ult i64 %j.next, 100
  br i1 %inner.cond, label %inner, label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %outer.cond = icmp ult i64 %i.next, %n
  br i1 %outer.cond, label %outer, label %exit, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
