;; Test that we do not assert/crash in distributed ThinLTO when an alias is
;; imported as a copy of the aliasee, but the aliasee summary is not in the
;; distributed index.

; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc src1.ll >src1.o
; RUN: opt -thinlto-bc src2.ll >src2.o
; RUN: llvm-lto2 run src1.o src2.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=src1.o,main,plx \
; RUN:  -r=src1.o,alias, \
; RUN:  -r=src2.o,alias,pl \
; RUN:  -r=src2.o,aliasee,pl \
; RUN:  -r=src2.o,_Znam, \
; RUN:  -o %t.out

;; ThinLTO backend for the importing module. Make sure this succeeds.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=src1.o.thinlto.bc \
; RUN:  -memprof-import-summary=src1.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -stats -pass-remarks=memprof-context-disambiguation \
; RUN:  src1.o -S 2>&1 | FileCheck %s

; CHECK: define available_externally hidden ptr @alias(ptr %{{.*}})

;--- src1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %call = call ptr @alias(ptr null), !callsite !0
  ret i32 0
}

declare ptr @alias(ptr)

!0 = !{i64 8632435727821051414}

;--- src2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@alias = hidden alias ptr (ptr), ptr @aliasee

define ptr @aliasee(ptr %0) !guid !1 {
entry:
  %call = call ptr @_Znam(i64 10), !memprof !2, !callsite !7
  ret ptr null
}

declare ptr @_Znam(i64)

!1 = !{i64 6174544574081827517}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
