;; Test that contexts trimmed at a stack id within a call's inlined stack id
;; sequence are also assigned to that call (unless disabled with
;; -memprof-assign-trimmed-inlined-contexts=false).
;;
;; Original code looks like:
;;
;; char *foo() { return new char[10]; }
;; char *foo2() { return new char[10]; }
;; char *baz() { return foo(); }
;; void bar() { baz(); foo2(); }
;; void mid() { bar(); }
;; void Y() { mid(); }
;; void E() { mid(); }
;; void Z() { bar(); }
;;
;; The allocation in foo is cold whenever it is reached via mid, and not cold
;; via Z. Because the behavior is unambiguous from mid's frame on, the cold
;; context is trimmed to end at mid's call to bar: [foo, baz, bar, mid].
;;
;; The allocation in foo2 is cold via E and not cold via Y, so its contexts
;; extend beyond mid's frame, creating context nodes for both mid's call to bar
;; and the callers of mid.
;;
;; In the IR, mid has been inlined into E (but not into Y), so E's call to bar
;; has the inlined stack ids [mid->bar, E->mid]. Since both have context nodes,
;; E's call is only assigned contexts that include E->mid, i.e. the foo2
;; contexts. The trimmed cold foo context is only assigned to the out-of-line
;; call in mid, whose only stack id is mid->bar. Without the new handling, E
;; calls the original bar and therefore the not cold foo allocation, even though
;; the profile says all contexts for foo through mid are cold.
;;
;; With the new handling, E's call is also given a duplicate of the trimmed foo
;; context, and E is assigned to call a clone of bar that calls a cold clone of
;; foo.
;;
;; Additionally, baz has been inlined into bar, so bar's call to foo has the
;; inlined stack ids [baz->foo, bar->baz], and is assigned the foo contexts
;; before the duplicate is created for E's call. This tests that the duplicate
;; is also given to bar's call, otherwise it would be left on the call-less
;; context nodes for baz->foo and bar->baz.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -memprof-assign-trimmed-inlined-contexts=false \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefixes=NOASSIGN \
; RUN:  --implicit-check-not="trimmed within"

; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  %s -S 2>&1 | FileCheck %s --check-prefixes=ASSIGN

;; Without the new handling, E calls the original bar, which calls the not cold
;; original foo.
; NOASSIGN: call in clone mid assigned to call function clone bar.memprof.1{{$}}
; NOASSIGN: call in clone bar.memprof.1 assigned to call function clone foo.memprof.1{{$}}
; NOASSIGN: call in clone foo.memprof.1 marked with memprof allocation attribute cold
; NOASSIGN: call in clone Z assigned to call function clone bar{{$}}
; NOASSIGN: call in clone bar assigned to call function clone foo{{$}}
; NOASSIGN: call in clone foo marked with memprof allocation attribute notcold
; NOASSIGN: call in clone E assigned to call function clone bar{{$}}
; NOASSIGN: define void @E()
; NOASSIGN-NEXT: entry:
; NOASSIGN-NEXT:   call void @bar()

;; With the new handling (the default), E calls a clone of bar that calls the
;; cold clone of foo (and the cold clone of foo2).
; ASSIGN: call in clone mid assigned to call function clone bar.memprof.1{{$}}
; ASSIGN: call in clone bar.memprof.1 assigned to call function clone foo.memprof.1{{$}}
; ASSIGN: call in clone E assigned to call function clone bar.memprof.2{{$}}
; ASSIGN: call in clone bar.memprof.2 assigned to call function clone foo.memprof.1{{$}}
; ASSIGN: call in clone foo.memprof.1 marked with memprof allocation attribute cold
; ASSIGN: call in clone Z assigned to call function clone bar{{$}}
; ASSIGN: call in clone bar assigned to call function clone foo{{$}}
; ASSIGN: call in clone foo marked with memprof allocation attribute notcold
; ASSIGN: call in clone bar.memprof.2 assigned to call function clone foo2.memprof.1{{$}}
; ASSIGN: call in clone foo2.memprof.1 marked with memprof allocation attribute cold
; ASSIGN: define void @E()
; ASSIGN-NEXT: entry:
; ASSIGN-NEXT:   call void @bar.memprof.2()
; ASSIGN: define void @bar.memprof.2()
; ASSIGN-NEXT: entry:
; ASSIGN-NEXT:   %call = call ptr @foo.memprof.1()
; ASSIGN-NEXT:   %call1 = call ptr @foo2.memprof.1()
; ASSIGN: 1 memprof-context-disambiguation - Number of calls assigned contexts trimmed within their inlined stack id sequence
; ASSIGN: 1 memprof-context-disambiguation - Number of (duplicated) context ids for contexts trimmed within a call's inlined stack id sequence

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !0, !callsite !5
  ret ptr %call
}

define ptr @foo2() {
entry:
  %call = call ptr @_Znam(i64 10) #0, !memprof !6, !callsite !11
  ret ptr %call
}

define void @bar() {
entry:
  %call = call ptr @foo(), !callsite !12
  %call1 = call ptr @foo2(), !callsite !13
  ret void
}

define void @mid() {
entry:
  call void @bar(), !callsite !14
  ret void
}

define void @Y() {
entry:
  call void @mid(), !callsite !15
  ret void
}

define void @E() {
entry:
  call void @bar(), !callsite !16
  ret void
}

define void @Z() {
entry:
  call void @bar(), !callsite !17
  ret void
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }

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
