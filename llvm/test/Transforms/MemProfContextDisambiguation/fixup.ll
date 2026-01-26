;; Test fixup of largest cold contexts.

;; This case has multiple recursive cycles in the cold context, which can be
;; made non-recursive with the inlining in the code.

;; -stats requires asserts
; REQUIRES: asserts

;; First try disabling detection of the largest cold contexts.
;; We will not get any cloning.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-top-n-important=0 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUNL	-pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --implicit-check-not="created clone" \
; RUN:	--implicit-check-not="Number of cold static allocations" \
; RUN:	--implicit-check-not="Number of function clones" \
; RUN:	--implicit-check-not="Number of important context ids" \
; RUN:	--implicit-check-not="Number of fixup"

;; Allow default detection of the largest cold contexts, but disable fixup.
;; We should find 1 important context, but still not get cloning.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-fixup-important=false \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUNL	-pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=TOPN1-NOFIXUP \
; RUN:	--implicit-check-not="created clone" \
; RUN:	--implicit-check-not="Number of cold static allocations" \
; RUN:	--implicit-check-not="Number of function clones" \
; RUN:	--implicit-check-not="Number of fixup"

; TOPN1-NOFIXUP: 1 memprof-context-disambiguation - Number of important context ids

;; Allow default detection of largest cold contexts, fixup is enabled by default.
;; This case should get fixup and cloning.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:	-pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefix=TOPN1

; TOPN1: created clone E.memprof.1
; TOPN1: created clone DB.memprof.1
; TOPN1: created clone CB.memprof.1
; TOPN1: created clone A.memprof.1
; TOPN1: call in clone main assigned to call function clone A.memprof.1
; TOPN1: call in clone A.memprof.1 assigned to call function clone CB.memprof.1
; TOPN1: call in clone CB.memprof.1 assigned to call function clone DB.memprof.1
; TOPN1: call in clone DB.memprof.1 assigned to call function clone E.memprof.1
; TOPN1: call in clone E.memprof.1 marked with memprof allocation attribute cold
; TOPN1: call in clone E marked with memprof allocation attribute notcold

; TOPN1: 1 memprof-context-disambiguation - Number of contexts with fixed edges
; TOPN1: 2 memprof-context-disambiguation - Number of fixup edges added
; TOPN1: 1 memprof-context-disambiguation - Number of important context ids

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @E() {
entry:
  %call = tail call ptr @_Znam(i64 10), !memprof !7, !callsite !14
  ret void
}

define void @DB() {
entry:
  tail call void @E(), !callsite !17
  ret void
}

define void @CB() {
entry:
  tail call void @DB(), !callsite !22
  ret void
}

define void @A() {
entry:
  tail call void @CB(), !callsite !20
  ret void
}

define i32 @main() {
entry:
  tail call void @A(), !callsite !25
  tail call void @A(), !callsite !27
  ret i32 0
}

declare ptr @_Znam(i64)

!7 = !{!8, !10}
!8 = !{!9, !"cold", !2}
!9 = !{i64 123, i64 234, i64 345, i64 234, i64 456, i64 234, i64 567, i64 678}
!2 = !{i64 12345, i64 200}
!10 = !{!11, !"notcold", !3}
!3 = !{i64 23456, i64 200}
!11 = !{i64 123, i64 234, i64 345, i64 234, i64 456, i64 234, i64 567, i64 789}
!14 = !{i64 123}
!17 = !{i64 234, i64 345}
!22 = !{i64 234, i64 456}
!20 = !{i64 234, i64 567}
!25 = !{i64 678}
!27 = !{i64 789}
