; Generate bitcode files with split LTO modules
; RUN: opt -unified-lto -thinlto-bc -thinlto-split-lto-unit -o %t0.bc %s
; RUN: llvm-dis -o %t3.ll %t0.bc
; RUN: FileCheck <%t3.ll.0 --check-prefix=M0 %s
; RUN: FileCheck <%t3.ll.1 --check-prefix=M1 %s
; RUN: llvm-bcanalyzer -dump %t0.bc | FileCheck --check-prefix=BCA0 %s

; ERROR: llvm-modextract: error: module index out of range; bitcode file contains 1 module(s)

; BCA0: <GLOBALVAL_SUMMARY_BLOCK
; BCA0: <FULL_LTO_GLOBALVAL_SUMMARY_BLOCK
; 16 = not eligible to import

$g = comdat any

@g = global i8 42, comdat, !type !0

; M0: define ptr @f()
define ptr @f() {
  ret ptr @g
}

; M1: !0 = !{i32 0, !"typeid"}
!0 = !{i32 0, !"typeid"}
