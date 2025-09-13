;; Make sure we assign the original callsite to a function clone (which will be
;; the original function clone), even when we cannot update its caller (due to
;; missing metadata e.g. from mismatched profiles). Otherwise we will try to use
;; the original function for a different clone, leading to confusion later when
;; rewriting the calls.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:		-memprof-verify-ccg -memprof-verify-nodes -stats -debug \
; RUN: 		-pass-remarks=memprof-context-disambiguation %s -S 2>&1 | \
; RUN:	FileCheck %s --implicit-check-not="Mismatch in call clone assignment" \
; RUN:		--implicit-check-not="Number of callsites assigned to call multiple non-matching clones"


; ModuleID = '<stdin>'
source_filename = "reduced.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: define void @A()
define void @A() {
  ; CHECK: call void @C()
  call void @C()
  ret void
}

; CHECK-LABEL: define void @B()
define void @B() {
  ; CHECK: call void @C.memprof.1()
  call void @C(), !callsite !1
  ret void
}

; CHECK-LABEL: define void @C()
define void @C() {
  ; CHECK: call void @F()
  call void @F(), !callsite !16
  ; CHECK: call void @D()
  call void @D(), !callsite !2
  ret void
}

; CHECK-LABEL: define void @D()
define void @D() {
  ; CHECK: call void @E()
  call void @E(), !callsite !3
  ; CHECK: call void @G()
  call void @G(), !callsite !17
  ret void
}

; CHECK-LABEL: define void @E()
define void @E() {
  ; CHECK: call ptr @_Znwm(i64 0) #[[NOTCOLD:[0-9]+]]
  %1 = call ptr @_Znwm(i64 0), !memprof !4, !callsite !9
  ret void
}

; CHECK-LABEL: define void @F()
define void @F() {
  ; CHECK: call void @G()
  call void @G(), !callsite !17
  ret void
}

; CHECK-LABEL: define void @G()
define void @G() {
  ; CHECK: call ptr @_Znwm(i64 0) #[[NOTCOLD]]
  %2 = call ptr @_Znwm(i64 0), !memprof !10, !callsite !15
  ret void
}

; CHECK-LABEL: define void @A1()
define void @A1() {
  ; CHECK: call void @C()
  call void @C(), !callsite !18
  ret void
}

; CHECK-LABEL: define void @B1()
define void @B1() {
  ; CHECK: call void @C.memprof.1()
  call void @C(), !callsite !19
  ret void
}

; CHECK-LABEL: define void @C.memprof.1()
  ; CHECK: call void @F.memprof.1()
  ; CHECK: call void @D.memprof.1()

; CHECK-LABEL: define void @D.memprof.1()
  ; CHECK: call void @E.memprof.1()
  ; CHECK: call void @G()

; CHECK-LABEL: define void @E.memprof.1()
  ; CHECK: call ptr @_Znwm(i64 0) #[[COLD:[0-9]+]]

; CHECK-LABEL: define void @F.memprof.1()
  ; CHECK: call void @G.memprof.1()

; CHECK-LABEL: define void @G.memprof.1()
  ; CHECK: call ptr @_Znwm(i64 0) #[[COLD]]

declare ptr @_Znwm(i64)

; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }

!0 = !{i64 123}
!1 = !{i64 234}
!2 = !{i64 345}
!3 = !{i64 456}
!4 = !{!5, !7}
!5 = !{!6, !"notcold"}
!6 = !{i64 567, i64 456, i64 345, i64 123}
!7 = !{!8, !"cold"}
!8 = !{i64 567, i64 456, i64 345, i64 234}
!9 = !{i64 567}
!10 = !{!11, !13}
!11 = !{!12, !"notcold"}
!12 = !{i64 678, i64 891, i64 789, i64 912}
!13 = !{!14, !"cold"}
!14 = !{i64 678, i64 891, i64 789, i64 812}
!15 = !{i64 678}
!16 = !{i64 789}
!17 = !{i64 891}
!18 = !{i64 912}
!19 = !{i64 812}
