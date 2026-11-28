;; Test that the ThinLTO search for frames missing from the profiled contexts
;; due to tail calls finds a tail call through an alias of the profiled callee.
;; Only the aliasee's function summary has a node in the graph, while the
;; wrapper's summary call edge is to the alias, so the search must resolve the
;; alias to compare it against the profiled callee.
;;
;; The original code looks like:
;;
;; char *foo_impl() { return new char[10]; }
;; char *foo() __attribute__((alias("foo_impl")));
;; char *bar() { return foo(); } // Tail calls the alias, frame missing.
;; int main() {
;;   bar(); // Not cold.
;;   bar(); // Cold.
;; }

; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o

; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z8foo_implv,plx \
; RUN:  -r=%t.o,_Z3foov,plx \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefixes=REMARKS,STATS
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR

;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,_Z8foo_implv,plx \
; RUN:  -r=%t.o,_Z3foov,plx \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=STATS

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S -o %t.be.ll 2>&1 | FileCheck %s --check-prefix=REMARKS
; RUN: FileCheck %s --check-prefix=IR < %t.be.ll

; REMARKS-DAG: created clone _Z8foo_implv.memprof.1
; REMARKS-DAG: created clone _Z3barv.memprof.1
; REMARKS-DAG: call in clone _Z3barv.memprof.1 assigned to call function clone _Z8foo_implv.memprof.1
; REMARKS-DAG: call in clone main assigned to call function clone _Z3barv.memprof.1

;; The missing frame is found once for each of the 2 profiled callers (main's
;; calls to bar).
; STATS: 2 memprof-context-disambiguation - Number of profiled callees found via tail calls

source_filename = "memprof-tailcall-alias.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_Z3foov = alias ptr (), ptr @_Z8foo_implv

; IR-LABEL: @_Z8foo_implv()
define ptr @_Z8foo_implv() #0 {
entry:
  ; IR: call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
  %call = tail call ptr @_Znam(i64 10) #2, !memprof !0, !callsite !5
  ret ptr %call
}

; Function Attrs: nobuiltin allocsize(0)
declare ptr @_Znam(i64) #1

;; The tail call to the alias foo is missing from the profiled contexts, which
;; have main calling foo_impl directly.
; IR-LABEL: @_Z3barv()
define ptr @_Z3barv() #0 {
entry:
  ;; The original version still calls the original foo_impl (via the alias,
  ;; before optimization).
  ; IR: tail call {{.*}} @_Z{{3foov|8foo_implv}}()
  %call = tail call ptr @_Z3foov()
  ret ptr %call
}

; IR-LABEL: @main()
define i32 @main() #0 {
  ;; The first call to bar is part of a not cold context, and should use the
  ;; original functions.
  ; IR: call {{.*}} @_Z3barv()
  %call = call ptr @_Z3barv(), !callsite !6
  ;; The second call to bar is part of a cold context, and should call the
  ;; cloned functions.
  ; IR: call {{.*}} @_Z3barv.memprof.1()
  %call1 = call ptr @_Z3barv(), !callsite !7
  ret i32 0
}

; IR-LABEL: @_Z8foo_implv.memprof.1()
; IR: call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
; IR-LABEL: @_Z3barv.memprof.1()
; IR: tail call {{.*}} @_Z8foo_implv.memprof.1()

; IR: attributes #[[NOTCOLD]] = { builtin allocsize(0) "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin allocsize(0) "memprof"="cold" }

attributes #0 = { noinline }
attributes #1 = { nobuiltin allocsize(0) }
attributes #2 = { builtin allocsize(0) }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 3186456655321080972, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 3186456655321080972, i64 -3421689549917153178}
!5 = !{i64 3186456655321080972}
!6 = !{i64 8632435727821051414}
!7 = !{i64 -3421689549917153178}
