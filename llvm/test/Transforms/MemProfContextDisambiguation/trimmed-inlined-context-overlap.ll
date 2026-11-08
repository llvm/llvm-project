;; Test that duplicates of contexts trimmed within a call's inlined stack id
;; sequence are not also given to a call whose stack id sequence overlaps that
;; call's, ending at a stack id within it. Otherwise, when there is no edge
;; between the nodes where the two sequences diverge, the overlapping call may
;; be processed first and move the duplicates to its new node, leaving the call
;; they were created for without them.
;;
;; Original code looks like:
;;
;; char *foo() { return new char[10]; }
;; void baz() { foo(); }
;; void bar() { baz(); foo(); }
;; void E() { bar(); }
;; void Y() { bar(); }
;; void Z() { baz(); }
;;
;; The allocation in foo is cold via bar's call to baz regardless of the caller
;; of bar, so that context is trimmed to end at bar's call to baz:
;; [foo, baz->foo, bar->baz]. It is not cold via Z's call to baz. For bar's
;; direct call to foo, it is cold via E and not cold via Y, so those contexts
;; extend to the callers of bar.
;;
;; In the IR, baz has been inlined into bar, and bar has been inlined into E.
;; E's call to foo through baz has the inlined stack ids
;; [baz->foo, bar->baz, E->bar]. All three have context nodes (the last due to
;; the contexts through bar's direct call to foo), but there is no edge between
;; the nodes for bar->baz and E->bar, since the trimmed context ends at the
;; former. E's call is therefore only assigned the trimmed cold context, as a
;; duplicate created for it. The out-of-line bar's call to foo through baz has
;; the inlined stack ids [baz->foo, bar->baz] and is assigned the original trimmed
;; context. It must not be given the duplicate as well: the node for bar->baz
;; is processed before the node for E->bar, so it would move the duplicate to
;; its new node, leaving E's call with no node and calling the not cold foo.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s

; CHECK: call in clone bar assigned to call function clone foo.memprof.1{{$}}
; CHECK: call in clone E assigned to call function clone foo.memprof.1{{$}}
; CHECK: call in clone E assigned to call function clone foo.memprof.1{{$}}
; CHECK: call in clone E assigned to call function clone foo.memprof.1{{$}}
; CHECK: call in clone foo.memprof.1 marked with memprof allocation attribute cold
; CHECK: call in clone baz assigned to call function clone foo{{$}}
; CHECK: call in clone bar assigned to call function clone foo{{$}}
; CHECK: call in clone foo marked with memprof allocation attribute notcold
; CHECK: define void @bar()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @foo.memprof.1()
; CHECK-NEXT:   %call1 = call ptr @foo()
; CHECK: define void @E()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @foo.memprof.1()
; CHECK-NEXT:   %call1 = call ptr @foo.memprof.1()
; CHECK-NEXT:   %call2 = call ptr @foo.memprof.1()
; CHECK: 1 memprof-context-disambiguation - Number of calls assigned contexts trimmed within their inlined stack id sequence
; CHECK: 1 memprof-context-disambiguation - Number of (duplicated) context ids for contexts trimmed within a call's inlined stack id sequence

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !9
  ret ptr %call
}

define void @baz() {
entry:
  %call = call ptr @foo(), !callsite !10
  ret void
}

define void @bar() {
entry:
  %call = call ptr @foo(), !callsite !11
  %call1 = call ptr @foo(), !callsite !12
  ret void
}

define void @E() {
entry:
  %call = call ptr @foo(), !callsite !13
  %call1 = call ptr @foo(), !callsite !14
  ;; A second call with the same stack ids as the first, which is only assigned
  ;; to the trimmed context, must be assigned to the same node as the first.
  %call2 = call ptr @foo(), !callsite !13
  ret void
}

define void @Y() {
entry:
  call void @bar(), !callsite !15
  ret void
}

define void @Z() {
entry:
  call void @baz(), !callsite !16
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = foo alloc, 2 = baz->foo, 3 = bar->baz, 4 = E->bar,
;; 5 = bar->foo, 6 = Y->bar, 7 = Z->baz.
!0 = !{!1, !3, !5, !7}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 7}
!5 = !{!6, !"cold"}
!6 = !{i64 1, i64 5, i64 4}
!7 = !{!8, !"notcold"}
!8 = !{i64 1, i64 5, i64 6}
!9 = !{i64 1}
!10 = !{i64 2}
!11 = !{i64 2, i64 3}
!12 = !{i64 5}
!13 = !{i64 2, i64 3, i64 4}
!14 = !{i64 5, i64 4}
!15 = !{i64 6}
!16 = !{i64 7}
