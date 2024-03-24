;; Test to ensure that importing of cloning decisions does not assert when
;; the callsite context is longer than the MIB context.
;; FIXME: Presumably this happened as a result of inlining, but in theory the
;; metadata should have been replaced with an attribute in that case. Need to
;; investigate why this is occuring.

; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:	-supports-hot-cold-new \
; RUN:	-r=%t.o,main,plx \
; RUN:	-r=%t.o,_Znam, \
; RUN:	-save-temps \
; RUN:	-o %t.out

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR
;; The call to new is not changed in this case.
; IR: call ptr @_Znam(i64 0)

source_filename = "memprof-import-fix.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !3
  ret i32 0
}

declare ptr @_Znam(i64)

attributes #0 = { noinline optnone }

!0 = !{!1}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951}
!3 = !{i64 9086428284934609951, i64 -5964873800580613432}
