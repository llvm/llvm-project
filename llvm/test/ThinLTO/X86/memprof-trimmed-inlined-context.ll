;; Test that contexts trimmed at a stack id within a call's inlined stack id
;; sequence are also assigned to that call (unless disabled with
;; -memprof-assign-trimmed-inlined-contexts=false), for the ThinLTO summary based
;; analysis. This is the same as the test in
;; Transforms/MemProfContextDisambiguation/trimmed-inlined-context.ll, see the
;; comments there for details.

; RUN: opt -thinlto-bc %s >%t.o

;; Without the new handling, E calls the original bar, which calls the not cold
;; original foo.
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -memprof-assign-trimmed-inlined-contexts=false \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -save-temps -o %t.out
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR-NOASSIGN

;; With the new handling (the default), E calls a clone of bar that calls the
;; cold clone of foo.
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=REMARKS-ASSIGN
; RUN: llvm-dis %t2.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR-ASSIGN

; IR-NOASSIGN: define internal {{.*}} @E()
; IR-NOASSIGN-NEXT: entry:
; IR-NOASSIGN-NEXT:   call {{.*}} @bar()

; REMARKS-ASSIGN: call in clone foo.memprof.1 marked with memprof allocation attribute cold
; REMARKS-ASSIGN: call in clone bar.memprof.2 assigned to call function clone foo.memprof.1{{$}}
; REMARKS-ASSIGN: call in clone E assigned to call function clone bar.memprof.2{{$}}
; IR-ASSIGN: define internal {{.*}} @E()
; IR-ASSIGN-NEXT: entry:
; IR-ASSIGN-NEXT:   call {{.*}} @bar.memprof.2()
; IR-ASSIGN: define internal {{.*}} @foo.memprof.1()
; IR-ASSIGN-NEXT: entry:
; IR-ASSIGN-NEXT:   call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR-ASSIGN: define internal {{.*}} @bar.memprof.2()
; IR-ASSIGN-NEXT: entry:
; IR-ASSIGN-NEXT:   call {{.*}} @foo.memprof.1()
; IR-ASSIGN-NEXT:   call {{.*}} @foo2.memprof.1()
; IR-ASSIGN: attributes #[[COLD]] = { builtin "memprof"="cold" }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal ptr @foo() #1 {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !5
  ret ptr %call
}

define internal ptr @foo2() #1 {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !6, !callsite !11
  ret ptr %call
}

define internal void @bar() #1 {
entry:
  %call = call ptr @foo(), !callsite !12
  %call1 = call ptr @foo2(), !callsite !13
  ret void
}

define internal void @mid() #1 {
entry:
  call void @bar(), !callsite !14
  ret void
}

define internal void @Y() #1 {
entry:
  call void @mid(), !callsite !15
  ret void
}

define internal void @E() #1 {
entry:
  call void @bar(), !callsite !16
  ret void
}

define internal void @Z() #1 {
entry:
  call void @bar(), !callsite !17
  ret void
}

define i32 @main() {
entry:
  call void @Y()
  call void @E()
  call void @Z()
  ret i32 0
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }
attributes #1 = { noinline optnone }

;; Stack ids: 1 = foo alloc, 2 = baz->foo, 3 = foo2 alloc, 4 = bar->foo2,
;; 5 = bar->baz, 10 = mid->bar, 11 = E->mid, 12 = Y->mid, 20 = Z->bar.
!0 = !{!1, !3}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 5, i64 10}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 5, i64 20}
!5 = !{i64 1}
!6 = !{!7, !9}
!7 = !{!8, !"cold"}
!8 = !{i64 3, i64 4, i64 10, i64 11}
!9 = !{!10, !"notcold"}
!10 = !{i64 3, i64 4, i64 10, i64 12}
!11 = !{i64 3}
!12 = !{i64 2, i64 5}
!13 = !{i64 4}
!14 = !{i64 10}
!15 = !{i64 12}
!16 = !{i64 10, i64 11}
!17 = !{i64 20}
