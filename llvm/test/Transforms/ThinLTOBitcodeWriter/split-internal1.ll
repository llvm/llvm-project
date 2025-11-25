; RUN: opt  -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract  -b -n 0 -o %t0 %t
; RUN: llvm-modextract  -b -n 1 -o %t1 %t
; RUN: not llvm-modextract -b -n 2 -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-dis  -o - %t0 | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis  -o - %t1 | FileCheck --check-prefix=M1 %s
; RUN: llvm-bcanalyzer  -dump %t0 | FileCheck --check-prefix=BCA0 %s
; RUN: llvm-bcanalyzer  -dump %t1 | FileCheck --check-prefix=BCA1 %s

; ERROR: llvm-modextract: error: module index out of range; bitcode file contains 2 module(s)

; BCA0: <GLOBALVAL_SUMMARY_BLOCK
; BCA1-NOT: <GLOBALVAL_SUMMARY_BLOCK

; M0: @g.581d7631532fa146ba4061179da39272 = external hidden global i8{{$}}
; M1: @g.581d7631532fa146ba4061179da39272 = hidden global i8 42, !type !0
@g = internal global i8 42, !type !0

; M0: define ptr @f()
; M1-NOT: @f()
define ptr @f() {
  ; M0: ret ptr @g.581d7631532fa146ba4061179da39272
  ret ptr @g
}

; M1: !0 = !{i32 0, !"typeid"}
!0 = !{i32 0, !"typeid"}
