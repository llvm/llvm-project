; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

target triple = "x86_64-unknown-linux-gnu"

@g = constant i32 0, !type !0
!0 = !{i32 0, !"typeid"}

; M0: module asm "ret"
; M1-NOT: module asm
module asm "ret"
