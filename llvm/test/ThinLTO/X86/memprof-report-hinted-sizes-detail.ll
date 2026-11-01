;; Test the -memprof-report-hinted-sizes-detail diagnostics for:
;; 1. Trimmed contexts whose caller above the trim point is cloned for another
;;    allocation: the report shows the post-cloning path from each clone of the
;;    outermost matched caller (here both mid clones reach the Cold clone of
;;    leafA through helper.memprof.1, so only one path is reported).
;; 2. Unmatched callsites (missing !callsite metadata on caller).
;; 3. Dead / unreferenced out-of-line allocation functions (e.g. when all
;;    callers inlined the allocation, leaving an unreferenced out-of-line copy).
;; 4. Address-taken allocation functions, which must not be reported as
;;    unreferenced.
;; 5. The "reaches" state reports the hint actually applied to the reached
;;    allocation clone (e.g. NotCold for an ambiguous allocation by default, or
;;    Cold with -memprof-cloning-cold-threshold), not the graph node's
;;    possibly ambiguous alloc types.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-report-hinted-sizes -memprof-report-hinted-sizes-detail \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefixes=ALL,LTO

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
; RUN:  -r=%t.o,leafD,plx \
; RUN:  -r=%t.o,callerD,plx \
; RUN:  -r=%t.o,fptr,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-report-hinted-sizes -memprof-report-hinted-sizes-detail \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefixes=ALL,THINLTO

;; Same as above but with a cloning cold byte threshold, so that the ambiguous
;; allocation in leafD is hinted cold.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-cloning-cold-threshold=80 \
; RUN:  -memprof-report-hinted-sizes -memprof-report-hinted-sizes-detail \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=PCT
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
; RUN:  -r=%t.o,leafD,plx \
; RUN:  -r=%t.o,callerD,plx \
; RUN:  -r=%t.o,fptr,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-cloning-cold-threshold=80 \
; RUN:  -memprof-report-hinted-sizes -memprof-report-hinted-sizes-detail \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=PCT

source_filename = "memprof-report-hinted-sizes-detail.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znam(i64)

@fptr = global ptr @addr_taken_alloc

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

;; Allocation whose contexts are only distinguished above an unmatched frame,
;; so it stays ambiguous (NotColdCold) after cloning.
define ptr @leafD() #0 {
entry:
  %p = call ptr @_Znam(i64 8), !memprof !40, !callsite !45
  ret ptr %p
}

define void @callerD() #0 {
entry:
  %p = call ptr @leafD(), !callsite !46
  ret void
}

;; Allocation function with no direct callers, but whose address is taken.
define internal ptr @addr_taken_alloc() #0 {
entry:
  %p = call ptr @_Znam(i64 4), !memprof !50, !callsite !53
  ret ptr %p
}

attributes #0 = { noinline }

!0 = !{!1, !3}
;; Context 101: cold context for leafA trimmed at mid -> helper (stack id 7)
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

!40 = !{!41, !43}
!41 = !{!42, !"cold", !47}
!42 = !{i64 41, i64 42, i64 43}
!43 = !{!44, !"notcold", !48}
!44 = !{i64 41, i64 42, i64 44}
!45 = !{i64 41}
!46 = !{i64 42}
!47 = !{i64 501, i64 9000}
!48 = !{i64 502, i64 1000}

!50 = !{!51}
!51 = !{!52, !"cold", !54}
!52 = !{i64 51, i64 52}
!53 = !{i64 51}
!54 = !{i64 601, i64 800}

; ALL-DAG: Cold full allocation context 101 with total size 1000 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafA.memprof.1; trimmed at frame 2 in mid; reaches Cold via mid -> helper.memprof.1 -> leafA.memprof.1]
; ALL-DAG: NotCold full allocation context 102 with total size 2000 is NotCold after cloning (internal context id {{[0-9]+}}) [alloc in leafA; matched 2 frames to other; reaches NotCold via other -> helper -> leafA]
; ALL-DAG: Cold full allocation context 201 with total size 3000 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafC.memprof.1; matched 3 frames to top1; reaches Cold via top1 -> mid.memprof.1 -> helper.memprof.2 -> leafC.memprof.1]
; ALL-DAG: NotCold full allocation context 202 with total size 4000 is NotCold after cloning (internal context id {{[0-9]+}}) [alloc in leafC; matched 3 frames to top2; reaches NotCold via top2 -> mid -> helper.memprof.1 -> leafC]
; ALL-DAG: Cold full allocation context 301 with total size 500 is NotColdCold after cloning (internal context id {{[0-9]+}}) [alloc in leaf_unmatched (no matched callers); unmatched callsite at frame 1/1 (stack id 22)]
;; dead_alloc_func is unreferenced: regular LTO has no dead symbol analysis, so
;; it is only reported as having no references; ThinLTO reports it dead.
; LTO-DAG: Cold full allocation context 401 with total size 700 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in dead_alloc_func (no references in IR/summary); unmatched callsite at frame 1/1 (stack id 32)]
; THINLTO-DAG: Cold full allocation context 401 with total size 700 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in dead_alloc_func (dead); unmatched callsite at frame 1/1 (stack id 32)]
; ALL-DAG: Cold full allocation context 501 with total size 9000 is NotColdCold after cloning (internal context id {{[0-9]+}}) [alloc in leafD; unmatched callsite at frame 2/2 (stack id 43); reaches NotCold via callerD -> leafD]
;; addr_taken_alloc is referenced (address taken), so it is not reported as
;; unreferenced; ThinLTO may have promoted/renamed the internal function.
; ALL-DAG: Cold full allocation context 601 with total size 800 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in addr_taken_alloc{{[^ ]*}} (no matched callers); unmatched callsite at frame 1/1 (stack id 52)]

; PCT-DAG: Cold full allocation context 501 with total size 9000 is NotColdCold after cloning marked Cold due to cold byte percent (internal context id {{[0-9]+}}) [alloc in leafD; unmatched callsite at frame 2/2 (stack id 43); reaches Cold via callerD -> leafD]
; PCT-DAG: NotCold full allocation context 502 with total size 1000 is NotColdCold after cloning marked Cold due to cold byte percent (internal context id {{[0-9]+}}) [alloc in leafD; unmatched callsite at frame 2/2 (stack id 44); reaches Cold via callerD -> leafD]
