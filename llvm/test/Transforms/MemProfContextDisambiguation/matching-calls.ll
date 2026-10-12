;; Test that matching calls (calls in the same function with identical callsite
;; metadata, recorded on a single context node) are all updated to call the
;; same callee version as the primary call in each version of their function.

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes %s -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %call = call noundef ptr @_Z3foov(), !callsite !0
  %call1 = call noundef ptr @_Z3foov(), !callsite !1
  ret i32 0
}

define internal ptr @_Z3barv() {
entry:
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #0, !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

;; Two calls to bar with the same callsite stack id (e.g. after duplication):
;; they are matching calls on a single context node.
define internal ptr @_Z3foov() {
entry:
  %call = call noundef ptr @_Z3barv(), !callsite !8
  %call2 = call noundef ptr @_Z3barv(), !callsite !8
  ret ptr null
}

attributes #0 = { builtin }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}

; CHECK: define internal {{.*}} @_Z3foov()
; CHECK-NEXT: entry:
; CHECK-NEXT:   call {{.*}} @_Z3barv()
; CHECK-NEXT:   call {{.*}} @_Z3barv()
; CHECK: define internal {{.*}} @_Z3foov.memprof.1()
; CHECK-NEXT: entry:
; CHECK-NEXT:   call {{.*}} @_Z3barv.memprof.1()
; CHECK-NEXT:   call {{.*}} @_Z3barv.memprof.1()
