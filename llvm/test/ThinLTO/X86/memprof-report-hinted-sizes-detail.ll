;; Test enhanced -memprof-report-hinted-sizes diagnostics for:
;; 1. Pruned contexts whose caller above the prune point is cloned for another
;;    allocation, causing one caller clone to reach the NotCold allocation clone
;;    even though the context node is assigned to the Cold clone.
;; 2. Unmatched callsites (missing !callsite metadata on caller).
;; 3. Dead / unreferenced out-of-line allocation functions (e.g. when all
;;    callers inlined the allocation, leaving an unreferenced out-of-line copy).

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-report-hinted-sizes %s -S 2>&1 | FileCheck %s --check-prefix=LTO

; RUN: opt -thinlto-bc -memprof-keep-context-size-info %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,top1,plx \
; RUN:  -r=%t.o,top2,plx \
; RUN:  -r=%t.o,other,plx \
; RUN:  -r=%t.o,mid,plx \
; RUN:  -r=%t.o,helper,plx \
; RUN:  -r=%t.o,leafA,plx \
; RUN:  -r=%t.o,leafC,plx \
; RUN:  -r=%t.o,unmatched_caller,plx \
; RUN:  -r=%t.o,leaf_unmatched,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-report-hinted-sizes \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=THINLTO

source_filename = "memprof-report-hinted-sizes-detail.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znam(i64)

define ptr @leafA() #0 {
entry:
  %p = call ptr @_Znam(i64 32), !memprof !0, !callsite !5
  ret ptr %p
}

define ptr @leafC() #0 {
entry:
  %p = call ptr @_Znam(i64 64), !memprof !6, !callsite !11
  ret ptr %p
}

define void @helper() #0 {
entry:
  %a = call ptr @leafA(), !callsite !12
  %c = call ptr @leafC(), !callsite !13
  ret void
}

define void @mid() #0 {
entry:
  call void @helper(), !callsite !14
  ret void
}

define void @other() #0 {
entry:
  call void @helper(), !callsite !15
  ret void
}

define void @top1() #0 {
entry:
  call void @mid(), !callsite !16
  ret void
}

define void @top2() #0 {
entry:
  call void @mid(), !callsite !17
  ret void
}

define ptr @leaf_unmatched() #0 {
entry:
  %p = call ptr @_Znam(i64 16), !memprof !22, !callsite !27
  ret ptr %p
}

define void @unmatched_caller() #0 {
entry:
  ;; Missing !callsite metadata blocks matching frame 1
  %p = call ptr @leaf_unmatched()
  ret void
}

define internal ptr @dead_alloc_func() #0 {
entry:
  %p = call ptr @_Znam(i64 128), !memprof !28, !callsite !31
  ret ptr %p
}

attributes #0 = { noinline }

!0 = !{!1, !3}
;; Context 101: cold context for leafA pruned at mid -> helper (stack id 7)
!1 = !{!2, !"cold", !18}
!2 = !{i64 1, i64 2, i64 7}
;; Context 102: notcold context for leafA via other -> helper (stack id 8)
!3 = !{!4, !"notcold", !19}
!4 = !{i64 1, i64 2, i64 8}
!5 = !{i64 1}

!6 = !{!7, !9}
;; Context 201: cold context for leafC via top1 -> mid -> helper
!7 = !{!8, !"cold", !20}
!8 = !{i64 5, i64 6, i64 7, i64 12}
;; Context 202: notcold context for leafC via top2 -> mid -> helper
!9 = !{!10, !"notcold", !21}
!10 = !{i64 5, i64 6, i64 7, i64 13}
!11 = !{i64 5}

!12 = !{i64 2}
!13 = !{i64 6}
!14 = !{i64 7}
!15 = !{i64 8}
!16 = !{i64 12}
!17 = !{i64 13}

!18 = !{i64 101, i64 1000}
!19 = !{i64 102, i64 2000}
!20 = !{i64 201, i64 3000}
!21 = !{i64 202, i64 4000}

!22 = !{!23, !25}
!23 = !{!24, !"cold", !32}
!24 = !{i64 21, i64 22}
!25 = !{!26, !"notcold", !33}
!26 = !{i64 21, i64 23}
!27 = !{i64 21}

!28 = !{!29}
!29 = !{!30, !"cold", !34}
!30 = !{i64 31, i64 32}
!31 = !{i64 31}

!32 = !{i64 301, i64 500}
!33 = !{i64 302, i64 600}
!34 = !{i64 401, i64 700}

; LTO-DAG: Cold full allocation context 101 with total size 1000 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafA.memprof.1; pruned at frame 2 in mid; reaches Cold via mid -> helper.memprof.1 -> leafA.memprof.1, NotCold via mid.memprof.1 -> helper -> leafA]
; LTO-DAG: NotCold full allocation context 102 with total size 2000 is NotCold after cloning (internal context id {{[0-9]+}}) [alloc in leafA; matched 2 frames to other; reaches NotCold via other -> helper -> leafA]
; LTO-DAG: Cold full allocation context 201 with total size 3000 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafC.memprof.1; matched 3 frames to top1; reaches Cold via top1 -> mid.memprof.1 -> helper -> leafC.memprof.1]
; LTO-DAG: NotCold full allocation context 202 with total size 4000 is NotCold after cloning (internal context id {{[0-9]+}}) [alloc in leafC; matched 3 frames to top2; reaches NotCold via top2 -> mid -> helper.memprof.1 -> leafC]
; LTO-DAG: Cold full allocation context 301 with total size 500 is NotColdCold after cloning (internal context id {{[0-9]+}}) [alloc in leaf_unmatched (no matched callers); unmatched callsite at frame 1/1 (stack id 22)]
; LTO-DAG: Cold full allocation context 401 with total size 700 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in dead_alloc_func (unreferenced); unmatched callsite at frame 1/1 (stack id 32)]

; THINLTO-DAG: Cold full allocation context 101 with total size 1000 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafA.memprof.1; pruned at frame 2 in mid; reaches Cold via mid -> helper.memprof.1 -> leafA.memprof.1, NotCold via mid.memprof.1 -> helper -> leafA]
; THINLTO-DAG: NotCold full allocation context 102 with total size 2000 is NotCold after cloning (internal context id {{[0-9]+}}) [alloc in leafA; matched 2 frames to other; reaches NotCold via other -> helper -> leafA]
; THINLTO-DAG: Cold full allocation context 201 with total size 3000 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafC.memprof.1; matched 3 frames to top1; reaches Cold via top1 -> mid.memprof.1 -> helper -> leafC.memprof.1]
; THINLTO-DAG: NotCold full allocation context 202 with total size 4000 is NotCold after cloning (internal context id {{[0-9]+}}) [alloc in leafC; matched 3 frames to top2; reaches NotCold via top2 -> mid -> helper.memprof.1 -> leafC]
; THINLTO-DAG: Cold full allocation context 301 with total size 500 is NotColdCold after cloning (internal context id {{[0-9]+}}) [alloc in leaf_unmatched (no matched callers); unmatched callsite at frame 1/1 (stack id 22)]
; THINLTO-DAG: Cold full allocation context 401 with total size 700 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in dead_alloc_func (dead); unmatched callsite at frame 1/1 (stack id 32)]
