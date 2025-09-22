;; Test to ensure assignments of calls to their callee function clones are
;; propagated when we create new callsite clones during function assignment.
;
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new %s -S | FileCheck %s

; ModuleID = 'funcassigncloning3.ll'
source_filename = "funcassigncloning3.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define void @A() {
  call void @_Znwm(), !memprof !0, !callsite !9
  ret void
}

define void @B() {
  call void @A(), !callsite !10
  ret void
}

define void @C() {
  call void @E(), !callsite !11
  ret void
}

define void @D() {
  call void @E(), !callsite !12
  ret void
}

; Function Attrs: cold
define void @E() {
  call void @B(), !callsite !13
  call void @A(), !callsite !14
  ret void
}

;; Clone E.memprof.2 is eventually created to satisfy the necessary combination
;; of caller edges, which causes creation of a new clone of callsite for the
;; call to A. Earlier this was assigned to call A.memprof.1 and we need to
;; ensure that assignment is propagated.

; CHECK: define void @E.memprof.2()
; CHECK-NEXT: call void @B()
; CHECK-NEXT: call void @A.memprof.1()

declare void @_Znwm()

!0 = !{!1, !3, !5, !7}
!1 = !{!2, !"cold"}
!2 = !{i64 761518489666860826, i64 -1420336805534834351, i64 -2943078617660248973, i64 3500755695426091485, i64 4378935957859808257, i64 4501820981166842392, i64 -6517003774656365154, i64 -3601339536116888955, i64 1856492280661618760, i64 5795517037440084991, i64 3898931366823636439}
!3 = !{!4, !"notcold"}
!4 = !{i64 761518489666860826, i64 -1420336805534834351, i64 -2943078617660248973, i64 3500755695426091485, i64 4378935957859808257, i64 4501820981166842392, i64 -6517003774656365154, i64 -3601339536116888955, i64 1856492280661618760, i64 5795517037440084991, i64 8489804099578214685}
!5 = !{!6, !"notcold"}
!6 = !{i64 761518489666860826, i64 -1420336805534834351, i64 -2943078617660248973, i64 3500755695426091485, i64 4378935957859808257, i64 4501820981166842392, i64 -3569839323322692552, i64 -4068062742094437340, i64 3898931366823636439}
!7 = !{!8, !"cold"}
!8 = !{i64 761518489666860826, i64 -1420336805534834351, i64 -2943078617660248973, i64 3500755695426091485, i64 4378935957859808257, i64 4501820981166842392, i64 -3569839323322692552, i64 -4068062742094437340, i64 8158446606478904094}
!9 = !{i64 761518489666860826, i64 -1420336805534834351, i64 -2943078617660248973, i64 3500755695426091485, i64 4378935957859808257, i64 4501820981166842392}
!10 = !{i64 -3569839323322692552}
!11 = !{i64 3898931366823636439}
!12 = !{i64 8158446606478904094}
!13 = !{i64 -4068062742094437340}
!14 = !{i64 -6517003774656365154, i64 -3601339536116888955, i64 1856492280661618760, i64 5795517037440084991}
