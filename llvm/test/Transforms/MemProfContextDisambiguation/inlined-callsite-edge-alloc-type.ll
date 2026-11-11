;; Test that the alloc type of an edge is updated when some of its context ids
;; are moved to a new node created for an inlined callsite. Otherwise the edge
;; keeps the alloc type it had before, and the allocation reached through it
;; may wrongly be treated as ambiguous rather than getting a specific type.
;;
;; Original code looks like:
;;
;; char *foo() { return new char[10]; }
;; void baz() { foo(); }
;; void bar() { baz(); }
;; void Z() { baz(); }
;;
;; The allocation is cold via bar and not cold via Z. In the IR, baz has been
;; inlined into bar, so bar's call to foo has the inlined stack ids
;; [baz->foo, bar->baz], and is matched to the cold context by creating a new
;; node, moving that context off of the edge from the allocation to the
;; baz->foo node. The not cold context remains on that edge, which is for the
;; out-of-line baz called from Z, so the original foo must be not cold (rather
;; than ambiguous) after cloning.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s

; CHECK: created clone foo.memprof.1
; CHECK: call in clone bar assigned to call function clone foo.memprof.1
; CHECK: call in clone foo.memprof.1 marked with memprof allocation attribute cold
; CHECK: call in clone baz assigned to call function clone foo
; CHECK: call in clone foo marked with memprof allocation attribute notcold
; CHECK: define ptr @foo()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
; CHECK: define ptr @foo.memprof.1()
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call ptr @_Znam(i64 10) #[[COLD:[0-9]+]]
; CHECK: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; CHECK: attributes #[[COLD]] = { builtin "memprof"="cold" }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !5
  ret ptr %call
}

define void @baz() {
entry:
  %call = call ptr @foo(), !callsite !6
  ret void
}

define void @bar() {
entry:
  %call = call ptr @foo(), !callsite !7
  ret void
}

define void @Z() {
entry:
  call void @baz(), !callsite !8
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = foo alloc, 2 = baz->foo, 3 = bar->baz, 4 = Z->baz.
!0 = !{!1, !3}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 3}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 4}
!5 = !{i64 1}
!6 = !{i64 2}
!7 = !{i64 2, i64 3}
!8 = !{i64 4}
