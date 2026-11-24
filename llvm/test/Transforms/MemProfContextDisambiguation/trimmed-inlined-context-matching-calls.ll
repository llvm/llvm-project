;; Test that a call with the same stack ids as an earlier call in the same
;; function is assigned to the node created for that call from only the
;; duplicates of contexts trimmed within their inlined stack id sequence, even
;; when a call processed before them was assigned all of the contexts on their
;; last stack id's node (after which no later call in that group can be assigned
;; any contexts from the graph).
;;
;; Original code looks like:
;;
;; char *wrap() { return new char[10]; }
;; char *foo() { return wrap(); }
;; void baz() { foo(); }
;; void bar() { baz(); foo(); }
;; void E() { bar(); }
;; void Y() { bar(); }
;; void Z() { baz(); }
;;
;; The allocation in wrap is cold via bar's call to baz regardless of the caller
;; of bar, so that context is trimmed to end at bar's call to baz:
;; [wrap, foo->wrap, baz->foo, bar->baz]. It is not cold via Z's call to baz.
;; For bar's direct call to foo, it is cold via E and not cold via Y, so those
;; contexts extend to the callers of bar.
;;
;; In the IR, baz and bar have been inlined into E, and foo has been inlined
;; into E's copy of bar's direct call. E has two identical calls to foo through
;; baz with the inlined stack ids [baz->foo, bar->baz, E->bar], which share a
;; node, and a call to wrap with the inlined stack ids [foo->wrap, bar->foo,
;; E->bar]. The latter sorts first among the calls ending at E->bar and is
;; assigned the only context on that node, the cold context through bar's
;; direct call. The calls through baz are then only assigned the trimmed cold
;; context, as a duplicate created for the first of them, and the second must
;; be assigned to the same node (otherwise it keeps calling the not cold foo).

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s

; CHECK: define void @E()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @foo.memprof.1()
; CHECK-NEXT:   %call1 = call ptr @wrap.memprof.1()
; CHECK-NEXT:   %call2 = call ptr @foo.memprof.1()
; CHECK: define ptr @wrap.memprof.1()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @_Znam(i64 10) #[[COLD:[0-9]+]]
; CHECK: define ptr @foo.memprof.1()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @wrap.memprof.1()
; CHECK: attributes #[[COLD]] = { builtin "memprof"="cold" }
; CHECK: 1 memprof-context-disambiguation - Number of calls assigned contexts trimmed within their inlined stack id sequence
; CHECK: 1 memprof-context-disambiguation - Number of (duplicated) context ids for contexts trimmed within a call's inlined stack id sequence

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @wrap() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !9
  ret ptr %call
}

define ptr @foo() {
entry:
  %call = call ptr @wrap(), !callsite !10
  ret ptr %call
}

define void @baz() {
entry:
  %call = call ptr @foo(), !callsite !11
  ret void
}

define void @bar() {
entry:
  %call = call ptr @foo(), !callsite !12
  %call1 = call ptr @foo(), !callsite !13
  ret void
}

define void @E() {
entry:
  %call = call ptr @foo(), !callsite !14
  %call1 = call ptr @wrap(), !callsite !15
  %call2 = call ptr @foo(), !callsite !14
  ret void
}

define void @Y() {
entry:
  call void @bar(), !callsite !16
  ret void
}

define void @Z() {
entry:
  call void @baz(), !callsite !17
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = wrap alloc, 2 = foo->wrap, 3 = baz->foo, 4 = bar->baz,
;; 5 = E->bar, 6 = bar->foo, 7 = Y->bar, 8 = Z->baz.
!0 = !{!1, !3, !5, !7}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3, i64 4}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 3, i64 8}
!5 = !{!6, !"cold"}
!6 = !{i64 1, i64 2, i64 6, i64 5}
!7 = !{!8, !"notcold"}
!8 = !{i64 1, i64 2, i64 6, i64 7}
!9 = !{i64 1}
!10 = !{i64 2}
!11 = !{i64 3}
!12 = !{i64 3, i64 4}
!13 = !{i64 6}
!14 = !{i64 3, i64 4, i64 5}
!15 = !{i64 2, i64 6, i64 5}
!16 = !{i64 7}
!17 = !{i64 8}
