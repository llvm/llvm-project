; RUN: not llvm-link %s %p/Inputs/appending-global.ll -S -o - 2>&1 | FileCheck %s
; RUN: not llvm-link %p/Inputs/appending-global.ll %s -S -o - 2>&1 | FileCheck %s

; Negative test to check that global variables with appending linkage
; and different address spaces cannot be linked.

; CHECK: error: Appending variables with different address spaces need to be linked!

@var = appending addrspace(1) global [ 1 x ptr ] undef
