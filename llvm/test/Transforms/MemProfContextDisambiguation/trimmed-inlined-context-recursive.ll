;; Test that contexts trimmed within a call's inlined stack id sequence are not
;; assigned to the call when a stack id in the sequence (before the last) has a
;; recursive context node, as such calls are not assigned a node at all.
;;
;; This is the same program as trimmed-inlined-context.ll, except that the cold
;; foo2 context passes through mid's call to bar twice, making that stack id's
;; node recursive (with -memprof-allow-recursive-callsites=false). E's call to
;; bar has the inlined stack ids [mid->bar, E->mid], so neither the foo2
;; contexts nor the cold foo context trimmed at mid->bar are assigned to it, and
;; it calls the original bar.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -memprof-allow-recursive-callsites=false \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --implicit-check-not="trimmed within" \
; RUN:  --implicit-check-not="created clone"

;; Nothing is cloned, as all cold contexts pass through the recursive node, so
;; both allocations remain ambiguous.
; CHECK: call in clone foo marked with memprof allocation attribute ambiguous
; CHECK: call in clone foo2 marked with memprof allocation attribute ambiguous
; CHECK: define void @E()
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @bar()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !5
  ret ptr %call
}

define ptr @foo2() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !6, !callsite !11
  ret ptr %call
}

define void @bar() {
entry:
  %call = call ptr @foo(), !callsite !12
  %call1 = call ptr @foo2(), !callsite !13
  ret void
}

define void @mid() {
entry:
  call void @bar(), !callsite !14
  ret void
}

define void @Y() {
entry:
  call void @mid(), !callsite !15
  ret void
}

define void @E() {
entry:
  call void @bar(), !callsite !16
  ret void
}

define void @Z() {
entry:
  call void @bar(), !callsite !17
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = foo alloc, 2 = baz->foo, 3 = foo2 alloc, 4 = bar->foo2,
;; 5 = bar->baz, 10 = mid->bar, 11 = E->mid, 12 = Y->mid, 20 = Z->bar.
!0 = !{!1, !3}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 5, i64 10}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 5, i64 20}
!5 = !{i64 1}
!6 = !{!7, !9}
!7 = !{!8, !"cold"}
!8 = !{i64 3, i64 4, i64 10, i64 10, i64 11}
!9 = !{!10, !"notcold"}
!10 = !{i64 3, i64 4, i64 10, i64 12}
!11 = !{i64 3}
!12 = !{i64 2, i64 5}
!13 = !{i64 4}
!14 = !{i64 10}
!15 = !{i64 12}
!16 = !{i64 10, i64 11}
!17 = !{i64 20}
