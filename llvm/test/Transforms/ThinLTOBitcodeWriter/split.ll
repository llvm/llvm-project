; Generate bitcode files with summary, as well as minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt -thinlto-bc -thin-link-bitcode-file=%t2 -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o %t0.bc %t
; RUN: llvm-modextract -b -n 1 -o %t1.bc %t
; RUN: llvm-modextract -b -n 0 -o %t0.thinlink.bc %t2
; RUN: llvm-modextract -b -n 1 -o %t1.thinlink.bc %t2
; RUN: not llvm-modextract -b -n 2 -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-dis -o - %t0.bc | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis -o - %t1.bc | FileCheck --check-prefix=M1 %s
; RUN: llvm-bcanalyzer -dump %t0.bc | FileCheck --check-prefix=BCA0 %s
; RUN: llvm-bcanalyzer -dump %t1.bc | FileCheck --check-prefix=BCA1 %s

; Make sure the combined index files produced by both the normal and the
; thin link bitcode files are identical
; RUN: llvm-lto -thinlto -o %t3 %t0.bc
; Copy the minimized bitcode to the regular bitcode path so the module
; paths in the index are the same.
; RUN: cp %t0.thinlink.bc %t0.bc
; RUN: llvm-lto -thinlto -o %t4 %t0.bc
; RUN: diff %t3.thinlto.bc %t4.thinlto.bc

; ERROR: llvm-modextract: error: module index out of range; bitcode file contains 2 module(s)

; BCA0: <GLOBALVAL_SUMMARY_BLOCK
; BCA1: <FULL_LTO_GLOBALVAL_SUMMARY_BLOCK
; 16 = not eligible to import
; BCA1: <PERMODULE_GLOBALVAR_INIT_REFS {{.*}} op1=16
; BCA1-NOT: <GLOBALVAL_SUMMARY_BLOCK

$g = comdat any

; M0: @g = external global i8{{$}}
; M1: @g = global i8 42, comdat, !type !0
@g = global i8 42, comdat, !type !0

; M0: define i8* @f()
; M1-NOT: @f()
define i8* @f() {
  ret i8* @g
}

; M1: !0 = !{i32 0, !"typeid"}
!0 = !{i32 0, !"typeid"}
