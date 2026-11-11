;; Test that a context trimmed at an interior callsite node is still identified
;; as such when a recursive context puts the same id on several callee and
;; caller edges of that node, so that the counts of ids on the callee and
;; caller edges no longer differ.
;;
;; Same as trimmed-context-clone.ll (the cold leafA context is trimmed at mid's
;; call to helper, and mid is cloned for top1 because of leafC), with recursion
;; added:
;;
;; void helper() { leafA(); leafC(); mid(); }
;; void mid() { helper(); mid(); }
;;
;; and a not cold leafC context via top2 going through mid->helper three times:
;; [leafC, helper->leafC, mid->helper, helper->mid, mid->helper, mid->mid,
;;  helper->mid, mid->helper, top2->mid]. At the mid->helper node it is on two
;; callee edges (from helper->leafC and helper->mid) but three caller edges
;; (from top2->mid, helper->mid and mid->mid). Recursive context cloning is
;; disabled so that it stays on the original mid->helper node, which then has
;; the same number of ids on its callee edges (trimmed leafA id, two leafC ids
;; via top2, the recursive id again) and its caller edges (two leafC ids via
;; top2, the recursive id twice more).
;;
;; As in trimmed-context-clone.ll, the mid clone for top1 must call a helper
;; clone calling the cold clone of leafA.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-clone-recursive-contexts=false \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=IR --check-prefix=STATS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @leafA() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !5
  ret ptr %call
}

define ptr @leafC() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !6, !callsite !11
  ret ptr %call
}

define void @helper() {
entry:
  %call = call ptr @leafA(), !callsite !12
  %call1 = call ptr @leafC(), !callsite !13
  call void @mid(), !callsite !18
  ret void
}

define void @mid() {
entry:
  call void @helper(), !callsite !14
  call void @mid(), !callsite !19
  ret void
}

define void @other() {
entry:
  call void @helper(), !callsite !15
  ret void
}

define void @top1() {
entry:
  call void @mid(), !callsite !16
  ret void
}

define void @top2() {
entry:
  call void @mid(), !callsite !17
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

;; Stack ids: 1 = leafA alloc, 2 = helper->leafA, 5 = leafC alloc,
;; 6 = helper->leafC, 7 = mid->helper, 8 = other->helper, 9 = helper->mid,
;; 10 = mid->mid, 12 = top1->mid, 13 = top2->mid.
!0 = !{!1, !3}
;; Cold context trimmed at mid's call to helper (no top1/top2 frame).
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 7}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 8}
!5 = !{i64 1}
!6 = !{!7, !9, !20}
!7 = !{!8, !"cold"}
!8 = !{i64 5, i64 6, i64 7, i64 12}
!9 = !{!10, !"notcold"}
!10 = !{i64 5, i64 6, i64 7, i64 13}
;; Recursive not cold context via top2.
!20 = !{!21, !"notcold"}
!21 = !{i64 5, i64 6, i64 7, i64 9, i64 7, i64 10, i64 9, i64 7, i64 13}
!11 = !{i64 5}
!12 = !{i64 2}
!13 = !{i64 6}
!14 = !{i64 7}
!15 = !{i64 8}
!16 = !{i64 12}
!17 = !{i64 13}
!18 = !{i64 9}
!19 = !{i64 10}

; IR: define {{.*}} @leafA()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
; IR: define {{.*}} @helper()
; IR:   call {{.*}} @leafA()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @mid()
; IR:   call {{.*}} @helper.memprof.1()
; IR: define {{.*}} @top1()
; IR:   call {{.*}} @mid.memprof.1()
; IR: define {{.*}} @top2()
; IR:   call {{.*}} @mid()
; IR: define {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @helper.memprof.1()
; IR:   call {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @helper.memprof.2()
; IR:   call {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @leafC.memprof.1()
; IR: define {{.*}} @mid.memprof.1()
; IR:   call {{.*}} @helper.memprof.2()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 1 memprof-context-disambiguation - Number of callsite clones without an edge to a callee's clones assigned the callee function clone of their original node
; STATS: 1 memprof-context-disambiguation - Number of context ids with call stacks trimmed at a callsite
