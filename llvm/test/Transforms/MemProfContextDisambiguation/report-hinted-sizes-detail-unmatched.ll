;; Test the -memprof-report-hinted-sizes-detail status for contexts whose
;; matching stopped at:
;; 1. A recursive callsite (a stack id repeated in the context, which is not
;;    matched when -memprof-allow-recursive-callsites=false).
;; 2. A callsite whose IR callee does not match the profiled callee (the call
;;    with the right !callsite id in callerE calls leafF, not leafE).

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-allow-recursive-callsites=false \
; RUN:  -memprof-report-hinted-sizes -memprof-report-hinted-sizes-detail \
; RUN:  %s -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znam(i64)

;; Allocation reached through a recursive callsite (stack id 62 repeats).
define ptr @leafR() #0 {
entry:
  %p = call ptr @_Znam(i64 24), !memprof !0, !callsite !3
  ret ptr %p
}

define ptr @recur(i32 %n) #0 {
entry:
  %p = call ptr @leafR(), !callsite !4
  ret ptr %p
}

;; Allocation whose profiled caller callsite (stack id 72) calls a different
;; function in the IR.
define ptr @leafE() #0 {
entry:
  %p = call ptr @_Znam(i64 48), !memprof !10, !callsite !13
  ret ptr %p
}

define ptr @leafF() #0 {
entry:
  ret ptr null
}

define void @callerE() #0 {
entry:
  %e = call ptr @leafE()
  %f = call ptr @leafF(), !callsite !14
  ret void
}

attributes #0 = { noinline }

!0 = !{!1}
!1 = !{!2, !"cold", !5}
!2 = !{i64 61, i64 62, i64 62}
!3 = !{i64 61}
!4 = !{i64 62}
!5 = !{i64 701, i64 300}

!10 = !{!11}
!11 = !{!12, !"cold", !15}
!12 = !{i64 71, i64 72}
!13 = !{i64 71}
!14 = !{i64 72}
!15 = !{i64 801, i64 400}

; CHECK-DAG: Cold full allocation context 701 with total size 300 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafR (no matched callers); unmatched recursive callsite at frame 1/1 (stack id 62)]
; CHECK-DAG: Cold full allocation context 801 with total size 400 is Cold after cloning (internal context id {{[0-9]+}}) [alloc in leafE (no matched callers); unmatched callsite (mismatched callee in callerE) at frame 1/1 (stack id 72)]
