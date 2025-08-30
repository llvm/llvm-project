;; Make sure we assign the original callsite to a function clone (which will be
;; the original function clone), even when we cannot update its caller (due to
;; missing metadata e.g. from mismatched profiles). Otherwise we will try to use
;; the original function for a different clone, leading to confusion later when
;; rewriting the calls.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,A,plx \
; RUN:  -r=%t.o,B,plx \
; RUN:  -r=%t.o,C,plx \
; RUN:  -r=%t.o,D,plx \
; RUN:  -r=%t.o,E,plx \
; RUN:  -r=%t.o,F,plx \
; RUN:  -r=%t.o,G,plx \
; RUN:  -r=%t.o,A1,plx \
; RUN:  -r=%t.o,B1,plx \
; RUN:  -r=%t.o,_Znwm, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -debug-only=memprof-context-disambiguation \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s \
; RUN:	--implicit-check-not="Mismatch in call clone assignment" \
; RUN:	--implicit-check-not="Number of callsites assigned to call multiple non-matching clones"

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR

; ModuleID = '<stdin>'
source_filename = "reduced.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; IR-LABEL: define dso_local void @A()
define void @A() #0 {
  ; IR: call void @C()
  call void @C()
  ret void
}

; IR-LABEL: define dso_local void @B()
define void @B() #0 {
  ; IR: call void @C.memprof.1()
  call void @C(), !callsite !1
  ret void
}

; IR-LABEL: define dso_local void @C()
define void @C() #0 {
  ; IR: call void @F()
  call void @F(), !callsite !16
  ; IR: call void @D()
  call void @D(), !callsite !2
  ret void
}

; IR-LABEL: define dso_local void @D()
define void @D() #0 {
  ; IR: call void @E()
  call void @E(), !callsite !3
  ; IR: call void @G()
  call void @G(), !callsite !17
  ret void
}

; IR-LABEL: define dso_local void @E()
define void @E() #0 {
  ; IR: call ptr @_Znwm(i64 0) #[[NOTCOLD:[0-9]+]]
  %1 = call ptr @_Znwm(i64 0), !memprof !4, !callsite !9
  ret void
}

; IR-LABEL: define dso_local void @F()
define void @F() #0 {
  ; IR: call void @G()
  call void @G(), !callsite !17
  ret void
}

; IR-LABEL: define dso_local void @G()
define void @G() #0 {
  ; IR: call ptr @_Znwm(i64 0) #[[NOTCOLD]]
  %2 = call ptr @_Znwm(i64 0), !memprof !10, !callsite !15
  ret void
}

; IR-LABEL: define dso_local void @A1()
define void @A1() #0 {
  ; IR: call void @C()
  call void @C(), !callsite !18
  ret void
}

; IR-LABEL: define dso_local void @B1()
define void @B1() #0 {
  ; IR: call void @C.memprof.1()
  call void @C(), !callsite !19
  ret void
}

; IR-LABEL: define dso_local void @C.memprof.1()
  ; IR: call void @F.memprof.1()
  ; IR: call void @D.memprof.1()

; IR-LABEL: define dso_local void @D.memprof.1()
  ; IR: call void @E.memprof.1()
  ; IR: call void @G()

; IR-LABEL: define dso_local void @E.memprof.1()
  ; IR: call ptr @_Znwm(i64 0) #[[COLD:[0-9]+]]

; IR-LABEL: define dso_local void @F.memprof.1()
  ; IR: call void @G.memprof.1()

; IR-LABEL: define dso_local void @G.memprof.1()
  ; IR: call ptr @_Znwm(i64 0) #[[COLD]]

declare ptr @_Znwm(i64)

attributes #0 = { noinline optnone }
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
