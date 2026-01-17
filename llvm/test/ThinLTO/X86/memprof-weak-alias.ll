;; Test handling of memprof ThinLTO when the call to update is a weak alias
;; where the prevailing weak aliasee is defined in a different module.

;; -stats requires asserts
; REQUIRES: asserts

;; Preparation steps to generate the bitcode and perform the thin link.
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc src1.ll >src1.o
; RUN: opt -thinlto-bc src2.ll >src2.o

;; First run the case where aliases (_Z8fooAliasv and _Z8fooAliasv2) and
;; aliasees (_Z3foov and _Z3foov2, respectively) are prevailing in the same
;; module.
; RUN: llvm-lto2 run src1.o src2.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:	-r=src1.o,main,plx \
; RUN:	-r=src1.o,_Znam, \
; RUN:	-r=src1.o,_Z8fooAliasv, \
; RUN:	-r=src1.o,_Z8fooAliasv2, \
; RUN:	-r=src1.o,_Z3foov,lx \
; RUN:	-r=src1.o,_Z3foov2,lx \
; RUN:	-r=src2.o,_Znam, \
; RUN:	-r=src2.o,_Z8fooAliasv,plx \
; RUN:	-r=src2.o,_Z8fooAliasv2,plx \
; RUN:	-r=src2.o,_Z3foov,plx \
; RUN:	-r=src2.o,_Z3foov2,plx \
; RUN:	-stats \
; RUN:	-o %t.out 2>&1 | FileCheck %s --implicit-check-not="prevailing in a different module"

;; Run ThinLTO backends. We should be able to clone correctly.
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-import-summary=src1.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  src1.o -S 2>&1 | FileCheck %s --check-prefix=IR-MOD1 --implicit-check-not=_Z3foov.memprof --implicit-check-not=_Z3foov2.memprof
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-import-summary=src2.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  src2.o -S 2>&1 | FileCheck %s --check-prefix=IR-MOD2

;; Now try where the aliasees are prevailing in the first module instead,
;; which only has the alias declarations.
; RUN: llvm-lto2 run src1.o src2.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:	-r=src1.o,main,plx \
; RUN:	-r=src1.o,_Znam, \
; RUN:	-r=src1.o,_Z8fooAliasv, \
; RUN:	-r=src1.o,_Z8fooAliasv2, \
; RUN:	-r=src1.o,_Z3foov,plx \
; RUN:	-r=src1.o,_Z3foov2,plx \
; RUN:	-r=src2.o,_Znam, \
; RUN:	-r=src2.o,_Z8fooAliasv,plx \
; RUN:	-r=src2.o,_Z8fooAliasv2,plx \
; RUN:	-r=src2.o,_Z3foov,lx \
; RUN:	-r=src2.o,_Z3foov2,lx \
; RUN:	-stats \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=STATS

;; We should have detected both aliasees and handle them conservatively.
; STATS: 2 memprof-context-disambiguation - Number of aliasees prevailing in a different module than its alias

;; Run ThinLTO backends. We should conservatively disable cloning.
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-import-summary=src1.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  src1.o -S 2>&1 | FileCheck %s --implicit-check-not "memprof.1"
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:	-memprof-import-summary=src2.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  src2.o -S 2>&1 | FileCheck %s --implicit-check-not "memprof.1"

;--- src1.ll
source_filename = "memprof-distrib-alias.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  ;; The first call to fooAlias does not allocate cold memory. It should call
  ;; the original function alias, which calls the original allocation decorated
  ;; with a "notcold" attribute.
  ; IR-MOD1:   call {{.*}} @_Z8fooAliasv()
  %call = call ptr @_Z8fooAliasv(), !callsite !0
  ;; The second call to fooAlias allocates cold memory. It should call the
  ;; cloned alias which calls a cloned allocation decorated with a "cold"
  ;; attribute.
  ; IR-MOD1:   call {{.*}} @_Z8fooAliasv.memprof.1()
  %call1 = call ptr @_Z8fooAliasv(), !callsite !1
  ;; The first call to fooAlias2 does not allocate cold memory. It should call
  ;; the original function alias, which calls the original allocation decorated
  ;; with a "notcold" attribute.
  ; IR-MOD1:   call {{.*}} @_Z8fooAliasv2()
  %call2 = call ptr @_Z8fooAliasv2(), !callsite !9
  ;; The second call to fooAlias2 allocates cold memory. It should call the
  ;; cloned alias which calls a cloned allocation decorated with a "cold"
  ;; attribute.
  ; IR-MOD1:   call {{.*}} @_Z8fooAliasv2.memprof.1()
  %call3 = call ptr @_Z8fooAliasv2(), !callsite !10
  ret i32 0
}

; IR-MOD1: declare {{.*}} @_Z8fooAliasv.memprof.1()
; IR-MOD1: declare {{.*}} @_Z8fooAliasv2.memprof.1()

declare ptr @_Znam(i64)

;; In this module the alias symbols are declarations. They alias with
;; weak_odr symbols _Z3foov and _Z3foov2 which are defined in both modules.
declare ptr @_Z8fooAliasv()
declare ptr @_Z8fooAliasv2()

define weak_odr ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

define weak_odr ptr @_Z3foov2() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !11, !callsite !16
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
!9 = !{i64 123}
!10 = !{i64 234}
!11 = !{!12, !14}
!12 = !{!13, !"notcold"}
!13 = !{i64 345, i64 123}
!14 = !{!15, !"cold"}
!15 = !{i64 345, i64 234}
!16 = !{i64 345}

;--- src2.ll
source_filename = "memprof-distrib-alias2.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_Znam(i64)

;; In this module the alias symbols are weak_odr aliases to weak_odr aliasees.
@_Z8fooAliasv = weak_odr alias ptr (...), ptr @_Z3foov
@_Z8fooAliasv2 = weak_odr alias ptr (...), ptr @_Z3foov2

define weak_odr ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

define weak_odr ptr @_Z3foov2() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !11, !callsite !16
  ret ptr null
}

; IR-MOD2: define weak_odr {{.*}} @_Z3foov()
; IR-MOD2:   call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
; IR-MOD2: define weak_odr {{.*}} @_Z3foov.memprof.1()
; IR-MOD2:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]
; IR-MOD2: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; IR-MOD2: attributes #[[COLD]] = { "memprof"="cold" }

attributes #0 = { noinline optnone }

!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!11 = !{!12, !14}
!12 = !{!13, !"notcold"}
!13 = !{i64 345, i64 123}
!14 = !{!15, !"cold"}
!15 = !{i64 345, i64 234}
!16 = !{i64 345}
