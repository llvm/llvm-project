; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-modextract -b -n 0 -o %t0 %t
; RUN: llvm-modextract -b -n 1 -o %t1 %t
; RUN: not llvm-modextract -b -n 2 -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=M1 %s
; RUN: llvm-bcanalyzer -dump %t0 | FileCheck --check-prefix=BCA0 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck --check-prefix=BCA1 %s

; ERROR: llvm-modextract: error: module index out of range; bitcode file contains 2 module(s)

; BCA0: <GLOBALVAL_SUMMARY_BLOCK
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

; M0: !llvm.dbg.cu
; M1-NOT: !llvm.dbg.cu
!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
