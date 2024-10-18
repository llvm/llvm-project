;; Test to ensure a call to a different callee but with the same debug info
;; (and therefore callsite metadata) as a preceding tail call in the alloc
;; context does not cause missing or incorrect cloning. This test is otherwise
;; the same as tailcall.ll.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -stats %s -S 2>&1 | FileCheck %s --check-prefix=STATS --check-prefix=IR

source_filename = "tailcall-aliased-location1.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local global [2 x ptr] [ptr @_Z2a1v, ptr @_Z2a2v], align 16

declare void @_Z2a1v() #0

declare void @_Z2a2v() #0

; Function Attrs: noinline
; IR-LABEL: @_Z3barv()
define ptr @_Z3barv() local_unnamed_addr #0 {
entry:
  ; IR: call ptr @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
  %call = tail call ptr @_Znam(i64 10) #2, !memprof !0, !callsite !5
  ret ptr %call
}

; Function Attrs: nobuiltin allocsize(0)
declare ptr @_Znam(i64) #1
declare void @blah()

; Function Attrs: noinline
; IR-LABEL: @_Z3bazv()
define ptr @_Z3bazv() #0 {
entry:
  ; IR: call ptr @_Z3barv()
  %call = tail call ptr @_Z3barv()
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @_Z3foov()
define ptr @_Z3foov() #0 {
entry:
  ; IR: call ptr @_Z3bazv()
  %call = tail call ptr @_Z3bazv()
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @main()
define i32 @main() #0 {
  ;; Preceding call to another callee but with the same debug location / callsite id
  call void @blah(), !callsite !6
  ;; The first call to foo is part of a cold context, and should use the
  ;; original functions.
  ;; allocation. The latter should call the cloned functions.
  ; IR: call ptr @_Z3foov()
  %call = tail call ptr @_Z3foov(), !callsite !6
  ;; The second call to foo is part of a cold context, and should call the
  ;; cloned functions.
  ; IR: call ptr @_Z3foov.memprof.1()
  %call1 = tail call ptr @_Z3foov(), !callsite !7
  %2 = load ptr, ptr @a, align 16
  call void %2(), !callsite !10
  ret i32 0
}

; IR-LABEL: @_Z3barv.memprof.1()
; IR: call ptr @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR-LABEL: @_Z3bazv.memprof.1()
; IR: call ptr @_Z3barv.memprof.1()
; IR-LABEL: @_Z3foov.memprof.1()
; IR: call ptr @_Z3bazv.memprof.1()

; IR: attributes #[[NOTCOLD]] = { builtin allocsize(0) "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin allocsize(0) "memprof"="cold" }

; STATS: 2 memprof-context-disambiguation - Number of profiled callees found via tail calls
; STATS: 4 memprof-context-disambiguation - Aggregate depth of profiled callees found via tail calls
; STATS: 2 memprof-context-disambiguation - Maximum depth of profiled callees found via tail calls

attributes #0 = { noinline }
attributes #1 = { nobuiltin allocsize(0) }
attributes #2 = { builtin allocsize(0) }

!0 = !{!1, !3, !8}
!1 = !{!2, !"notcold"}
!2 = !{i64 3186456655321080972, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 3186456655321080972, i64 -3421689549917153178}
!5 = !{i64 3186456655321080972}
!6 = !{i64 8632435727821051414}
!7 = !{i64 -3421689549917153178}
!8 = !{!9, !"notcold"}
!9 = !{i64 3186456655321080972, i64 1}
!10 = !{i64 1}
