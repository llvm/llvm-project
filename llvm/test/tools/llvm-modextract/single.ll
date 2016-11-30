; RUN: llvm-as -o %t %s
; RUN: llvm-modextract -n 0 -o - %t | llvm-dis | FileCheck %s
; RUN: not llvm-modextract -n 1 -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck %s
; RUN: not llvm-modextract -b -n 1 -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s

; CHECK: define void @f()
; ERROR: llvm-modextract: error: module index out of range; bitcode file contains 1 module(s)

define void @f() {
  ret void
}
