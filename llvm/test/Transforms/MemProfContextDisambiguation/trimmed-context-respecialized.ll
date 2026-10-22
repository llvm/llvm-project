;; Test that the callee version an implicit caller requires (see
;; trimmed-context-clone.ll) is kept when the function clone it was assigned to
;; is later specialized further for another callsite.
;;
;; As in trimmed-context-clone.ll, the cold leafA context is trimmed at mid's
;; call to helper, and mid is cloned for top1 because of leafC (cold via top1,
;; not cold via top2). A third allocation, leafE, is cold via other and via mid
;; from top2, and not cold via base:
;;
;; char *leafE() { return new char[10]; }
;; void helper() { leafE(); leafA(); leafC(); }
;; void base() { helper(); }
;;
;; Processing helper's callsites in order, leafE puts the original mid and other
;; into the same helper clone. At leafA, other keeps the original helper->leafA
;; node in that clone and the original mid conflicts, so it gets a further
;; helper clone with the cold leafA callsite clone; the mid clone for top1 (an
;; implicit caller: no edge to the leafA nodes, trimmed context applies to it)
;; is still unassigned and is assigned that same helper clone. At leafC the two
;; conflict again (mid calls the original leafC, its clone the cold one), and
;; the original mid moves on to yet another helper clone, taking the cold leafA
;; callsite clone with it and leaving an empty one behind. The leafA call in the
;; helper clone the mid clone kept must still follow the cold leafA callsite
;; clone (recorded when the implicit caller was assigned), and not fall back to
;; the version in the helper clone it was derived from (the original leafA).

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
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

define ptr @leafE() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !30, !callsite !37
  ret ptr %call
}

define void @helper() {
entry:
  %call0 = call ptr @leafE(), !callsite !38
  %call = call ptr @leafA(), !callsite !12
  %call1 = call ptr @leafC(), !callsite !13
  ret void
}

define void @mid() {
entry:
  call void @helper(), !callsite !14
  ret void
}

define void @other() {
entry:
  call void @helper(), !callsite !15
  ret void
}

define void @base() {
entry:
  call void @helper(), !callsite !39
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
;; 6 = helper->leafC, 7 = mid->helper, 8 = other->helper, 12 = top1->mid,
;; 13 = top2->mid, 20 = leafE alloc, 21 = helper->leafE, 22 = base->helper.
!0 = !{!1, !3}
;; Cold context trimmed at mid's call to helper (no top1/top2 frame).
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 7}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 8}
!5 = !{i64 1}
!6 = !{!7, !9}
!7 = !{!8, !"cold"}
!8 = !{i64 5, i64 6, i64 7, i64 12}
!9 = !{!10, !"notcold"}
!10 = !{i64 5, i64 6, i64 7, i64 13}
!11 = !{i64 5}
!12 = !{i64 2}
!13 = !{i64 6}
!14 = !{i64 7}
!15 = !{i64 8}
!16 = !{i64 12}
!17 = !{i64 13}
!30 = !{!31, !33, !35}
!31 = !{!32, !"cold"}
!32 = !{i64 20, i64 21, i64 7, i64 13}
!33 = !{!34, !"cold"}
!34 = !{i64 20, i64 21, i64 8}
!35 = !{!36, !"notcold"}
!36 = !{i64 20, i64 21, i64 22}
!37 = !{i64 20}
!38 = !{i64 21}
!39 = !{i64 22}

;; top1 reaches the cold leafA through the mid clone and the helper clone it
;; calls; top2 through the original mid and its helper clone.
; IR: define {{.*}} @leafA()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
; IR: define {{.*}} @helper()
; IR:   call {{.*}} @leafE()
; IR:   call {{.*}} @leafA()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @mid()
; IR:   call {{.*}} @helper.memprof.3()
; IR: define {{.*}} @other()
; IR:   call {{.*}} @helper.memprof.1()
; IR: define {{.*}} @base()
; IR:   call {{.*}} @helper()
; IR: define {{.*}} @top1()
; IR:   call {{.*}} @mid.memprof.1()
; IR: define {{.*}} @top2()
; IR:   call {{.*}} @mid()
; IR: define {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR: define {{.*}} @helper.memprof.1()
; IR:   call {{.*}} @leafE.memprof.1()
; IR:   call {{.*}} @leafA()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @helper.memprof.2()
; IR:   call {{.*}} @leafE
; IR:   call {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @leafC.memprof.1()
; IR: define {{.*}} @helper.memprof.3()
; IR:   call {{.*}} @leafE.memprof.1()
; IR:   call {{.*}} @leafA.memprof.1()
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @mid.memprof.1()
; IR:   call {{.*}} @helper.memprof.2()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 1 memprof-context-disambiguation - Number of callsite clones without an edge to a callee's clones assigned the callee function clone of their original node
; STATS: 1 memprof-context-disambiguation - Number of context ids with call stacks trimmed at a callsite
