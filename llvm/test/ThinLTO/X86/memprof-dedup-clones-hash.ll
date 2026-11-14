;; Test that ThinLTO backend clones that assign the same callee clone numbers to
;; different callsites are not treated as duplicates of each other.
;;
;; Original code looks like:
;;
;; char *foo() { return new char[10]; }
;; char *foo2() { return new char[10]; }
;; void bar() { foo(); foo2(); }
;; void A() { bar(); }
;; void B() { bar(); }
;;
;; The allocation in foo is cold via A and not cold via B, while the allocation
;; in foo2 is cold via B and not cold via A. Therefore bar must be cloned, with
;; one copy calling foo.memprof.1 and the original foo2, and the other calling
;; the original foo and foo2.memprof.1. In the summary these copies of bar have
;; callsite clone numbers [1, 0] and [0, 1], respectively. The ThinLTO backend
;; computes a hash of these to detect duplicate function clones, which it emits
;; as aliases instead. The hash previously included only the non-zero clone
;; numbers, without which callsite they belong to, so the two copies of bar were
;; considered duplicates.

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS \
; RUN:  --implicit-check-not="created clone alias"
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR

;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -o %t2.out

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR \
; RUN:  --check-prefix=REMARKS --implicit-check-not="created clone alias"

; REMARKS: created clone bar.memprof.1
; REMARKS-DAG: call in clone bar assigned to call function clone foo2.memprof.1
; REMARKS-DAG: call in clone bar.memprof.1 assigned to call function clone foo.memprof.1
; REMARKS-DAG: call in clone A assigned to call function clone bar.memprof.1

; IR: define internal {{.*}} @bar()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @foo()
; IR-NEXT:   call {{.*}} @foo2.memprof.1()
; IR: define internal {{.*}} @A()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @bar.memprof.1()
; IR: define internal {{.*}} @B()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @bar()
; IR: define internal {{.*}} @bar.memprof.1()
; IR-NEXT: entry:
; IR-NEXT:   call {{.*}} @foo.memprof.1()
; IR-NEXT:   call {{.*}} @foo2()

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

define internal void @A() #1 {
entry:
  call void @bar(), !callsite !14
  ret void
}

define internal void @B() #1 {
entry:
  call void @bar(), !callsite !15
  ret void
}

define i32 @main() {
entry:
  call void @A()
  call void @B()
  ret i32 0
}

declare ptr @_Znam(i64)

attributes #0 = { builtin }
attributes #1 = { noinline optnone }

;; Stack ids: 1 = foo alloc, 2 = bar->foo, 3 = foo2 alloc, 4 = bar->foo2,
;; 10 = A->bar, 20 = B->bar.
!0 = !{!1, !3}
!1 = !{!2, !"cold"}
!2 = !{i64 1, i64 2, i64 10}
!3 = !{!4, !"notcold"}
!4 = !{i64 1, i64 2, i64 20}
!5 = !{i64 1}
!6 = !{!7, !9}
!7 = !{!8, !"cold"}
!8 = !{i64 3, i64 4, i64 20}
!9 = !{!10, !"notcold"}
!10 = !{i64 3, i64 4, i64 10}
!11 = !{i64 3}
!12 = !{i64 2}
!13 = !{i64 4}
!14 = !{i64 10}
!15 = !{i64 20}
