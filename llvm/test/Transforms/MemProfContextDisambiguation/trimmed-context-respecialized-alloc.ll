;; Same as trimmed-context-respecialized.ll, with the leafA allocation placed
;; directly in helper, so that the callsite node the implicit caller (the mid
;; clone for top1) must follow is an allocation node. When the original mid is
;; specialized further at leafC, the cold allocation clone it called is moved
;; to its new helper clone and the clone recorded for the mid clone's helper
;; clone is left without contexts (and without an allocation type). The call
;; update must resolve the recorded source to the clone that now carries the
;; trimmed context instead of applying the empty clone's (None) allocation
;; type, so that the mid clone's helper clone still hints the allocation cold.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=IR --check-prefix=STATS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !12
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

;; Stack ids: 2 = allocation in helper, 5 = leafC alloc,
;; 6 = helper->leafC, 7 = mid->helper, 8 = other->helper, 12 = top1->mid,
;; 13 = top2->mid, 20 = leafE alloc, 21 = helper->leafE, 22 = base->helper.
!0 = !{!1, !3}
;; Cold context trimmed at mid's call to helper (no top1/top2 frame).
!1 = !{!2, !"cold"}
!2 = !{i64 2, i64 7}
!3 = !{!4, !"notcold"}
!4 = !{i64 2, i64 8}
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


;; top1 reaches the cold allocation through the mid clone and the helper clone it
;; calls (helper.memprof.2); top2 through the original mid and helper.memprof.3;
;; other through helper.memprof.1 (not cold); base through the original helper.
; IR: define {{.*}} @helper()
; IR:   call {{.*}} @leafE()
; IR:   call {{.*}} @_Znam(i64 10) #[[DEFAULT:[0-9]+]]
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
; IR: define {{.*}} @helper.memprof.1()
; IR:   call {{.*}} @leafE.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @helper.memprof.2()
; IR:   call {{.*}} @leafE()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR:   call {{.*}} @leafC.memprof.1()
; IR: define {{.*}} @helper.memprof.3()
; IR:   call {{.*}} @leafE.memprof.1()
; IR:   call {{.*}} @_Znam(i64 10) #[[COLD]]
; IR:   call {{.*}} @leafC()
; IR: define {{.*}} @mid.memprof.1()
; IR:   call {{.*}} @helper.memprof.2()
; IR: attributes #[[NOTCOLD]] = { builtin "memprof"="notcold" }
; IR: attributes #[[DEFAULT]] = { builtin }
; IR: attributes #[[COLD]] = { builtin "memprof"="cold" }

; STATS: 1 memprof-context-disambiguation - Number of callsite clones without an edge to a callee's clones assigned the callee function clone of their original node
; STATS: 1 memprof-context-disambiguation - Number of context ids with call stacks trimmed at a callsite
