;; Test that instantiated MemProf function clones (.memprof.N) strip !kcfi_type
;; metadata in both regular/non-ThinLTO and ThinLTO backends, while original
;; functions retain !kcfi_type so only original functions emit KCFI prefixes.

;; Non-ThinLTO / regular LTO module pass path.
; RUN: opt -passes=memprof-context-disambiguation -supports-hot-cold-new \
; RUN:   %s -S | FileCheck %s --check-prefix=IR --implicit-check-not="!kcfi_type"

;; ThinLTO path (including codegen check for __cfi_ KCFI prefix symbols).
; RUN: opt -thinlto-bc %s -o %t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:   -supports-hot-cold-new \
; RUN:   -r=%t.o,main,plx \
; RUN:   -r=%t.o,_Z3foov,plx \
; RUN:   -r=%t.o,_Z3barv,plx \
; RUN:   -r=%t.o,_Znam, \
; RUN:   -save-temps -o %t.out
; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR \
; RUN:   --implicit-check-not="!kcfi_type"
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=NM

;; Original functions retain !kcfi_type; instantiated .memprof.1 clones do not.
; IR-DAG: define {{.*}}ptr @_Z3barv() {{.*}}!kcfi_type ![[KCFI:[0-9]+]]{{.*}} {
; IR-DAG: define {{.*}}ptr @_Z3foov() {{.*}}!kcfi_type ![[KCFI]]{{.*}} {
; IR-DAG: define {{.*}}ptr @_Z3barv.memprof.1() {{.*}} {
; IR-DAG: define {{.*}}ptr @_Z3foov.memprof.1() {{.*}} {
; IR: ![[KCFI]] = !{i32 12345678}

;; Only original functions get __cfi_ KCFI type prefix symbols in the object.
; NM-NOT: __cfi_{{.*}}.memprof.
; NM-DAG: T __cfi__Z3barv
; NM-DAG: T __cfi__Z3foov
; NM-NOT: __cfi_{{.*}}.memprof.

source_filename = "memprof-kcfi.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !0
  %call1 = call ptr @_Z3foov(), !callsite !1
  ret i32 0
}

define ptr @_Z3barv() #0 !kcfi_type !10 {
entry:
  %call = call ptr @_Znam(i64 10) #1, !memprof !2, !callsite !7
  ret ptr %call
}

declare ptr @_Znam(i64)

define ptr @_Z3foov() #0 !kcfi_type !10 {
entry:
  %call = call ptr @_Z3barv(), !callsite !8
  ret ptr %call
}

attributes #0 = { noinline optnone }
attributes #1 = { builtin }

!llvm.module.flags = !{!9}

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 2732490490862098848}
!9 = !{i32 4, !"kcfi", i32 1}
!10 = !{i32 12345678}
