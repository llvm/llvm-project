;; Test that passing -supports-hot-cold-new to the LTO link allows context
;; disambiguation to proceed, and also prevents memprof metadata and attributes
;; from being removed from the LTO backend, and vice versa without passing
;; -supports-hot-cold-new.

;; Note that this tests regular LTO (with a summary) due to the module flag
;; disabling ThinLTO.

;; First check with -supports-hot-cold-new.
; RUN: opt -module-summary %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-dump-ccg \
; RUN:	-print-before=memprof-context-disambiguation \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=DUMP --check-prefix=IR

; IR: !memprof {{.*}} !callsite
; IR: "memprof"="cold"

; DUMP: Callsite Context Graph:

;; Next check without -supports-hot-cold-new, we should not perform
;; context disambiguation, and we should strip memprof metadata and
;; attributes before optimization.
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-dump-ccg \
; RUN:	-print-before=memprof-context-disambiguation \
; RUN:	-o %t.out 2>&1 | FileCheck %s --allow-empty \
; RUN:  --implicit-check-not "Callsite Context Graph:" \
; RUN: 	--implicit-check-not "!memprof" --implicit-check-not "!callsite" \
; RUN: 	--implicit-check-not "memprof"="cold"

;; Ensure the attributes and metadata are stripped when running a non-LTO pipeline.
; RUN: opt -O3 %t.o -S | FileCheck %s \
; RUN: 	--implicit-check-not "!memprof" --implicit-check-not "!callsite" \
; RUN: 	--implicit-check-not "memprof"="cold"

source_filename = "memprof-supports-hot-cold-new.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  %call2 = call ptr @_Znam(i64 0) #1
  ret i32 0
}

declare ptr @_Znam(i64)

attributes #0 = { noinline optnone }
attributes #1 = { "memprof"="cold" }

!llvm.module.flags = !{!6}

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!5 = !{i64 9086428284934609951}

;; Force regular LTO even though we have a summary.
!6 = !{i32 1, !"ThinLTO", i32 0}
