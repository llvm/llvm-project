;; Test handling of memprof distributed ThinLTO when the call to update is an alias.

;; Preparation steps to generate the bitcode and perform the thin link.
; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-o %t2.out

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  %t.o -S 2>&1 | FileCheck %s --check-prefix=IR

source_filename = "memprof-distrib-alias.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  ;; The first call to fooAlias does not allocate cold memory. It should call
  ;; the original function alias, which calls the original allocation decorated
  ;; with a "notcold" attribute.
  ; IR:   call {{.*}} @_Z8fooAliasv()
  %call = call ptr @_Z8fooAliasv(), !callsite !0
  ;; The second call to fooAlias allocates cold memory. It should call the
  ;; cloned function which calls a cloned allocation decorated with a "cold"
  ;; attribute.
  ; IR:   call {{.*}} @_Z3foov.memprof.1()
  %call1 = call ptr @_Z8fooAliasv(), !callsite !1
  ret i32 0
}

; IR: define internal {{.*}} @_Z3foov()
; IR:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IR: define internal {{.*}} @_Z3foov.memprof.1()
; IR:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR: attributes #[[COLD]] = { "memprof"="cold" }

declare ptr @_Znam(i64)

@_Z8fooAliasv = internal alias ptr (...), ptr @_Z3foov

define internal ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

attributes #0 = { noinline optnone }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
