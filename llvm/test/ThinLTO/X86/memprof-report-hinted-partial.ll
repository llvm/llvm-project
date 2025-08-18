;; Test that we get hinted size reporting for just the subset of MIBs that
;; contain context size info in the metadata.

;; Generate the bitcode including ThinLTO summary. Specify
;; -memprof-min-percent-max-cold-size (value doesn't matter) to indicate to
;; the bitcode writer that it should expect and optimize for partial context
;; size info.
; RUN: opt -thinlto-bc -memprof-min-percent-max-cold-size=50 %s >%t.o

; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-memprof-report-hinted-sizes \
; RUN:	-o %t.out 2>&1 | FileCheck %s --check-prefix=SIZES

;; We should only get these two messages from -memprof-report-hinted-sizes
;; as they are the only MIBs with recorded context size info.
; SIZES-NOT: full allocation context
; SIZES: Cold full allocation context 456 with total size 200 is Cold after cloning (context id 2)
; SIZES: Cold full allocation context 789 with total size 300 is Cold after cloning (context id 2)
; SIZES-NOT: full allocation context

source_filename = "memprof-report-hinted-partial.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Z3foov(), !callsite !0
  %call1 = call ptr @_Z3foov(), !callsite !1
  ret i32 0
}

define internal ptr @_Z3barv() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

define internal ptr @_Z3bazv() #0 {
entry:
  %call = call ptr @_Z3barv(), !callsite !8
  ret ptr null
}

define internal ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Z3bazv(), !callsite !9
  ret ptr null
}

; uselistorder directives
uselistorder ptr @_Z3foov, { 1, 0 }

attributes #0 = { noinline optnone }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5, !13}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!5 = !{!6, !"cold", !11, !12}
!6 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 -5964873800580613432}
!9 = !{i64 2732490490862098848}
!11 = !{i64 456, i64 200}
!12 = !{i64 789, i64 300}
!13 = !{!14, !"cold"}
!14 = !{i64 9086428284934609951, i64 12345}
