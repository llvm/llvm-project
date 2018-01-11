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

; M0: @g = external global void ()*{{$}}
; M1: @g = global void ()* @"f$13757e0fb71915e385efa4dc9d1e08fd", !type !0
@g = global void ()* @f, !type !0

; M0: define dso_local hidden void @"f$13757e0fb71915e385efa4dc9d1e08fd"()
; M1: declare dso_local hidden void @"f$13757e0fb71915e385efa4dc9d1e08fd"()
define internal void @f() {
  call void @f2()
  ret void
}

; M0: define internal void @f2()
define internal void @f2() {
  ret void
}

; M1: !0 = !{i32 0, !"typeid"}
!0 = !{i32 0, !"typeid"}
