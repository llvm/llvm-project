; RUN: not llvm-link %s %p/Inputs/appending-global.ll -S -o - 2>&1 | FileCheck %s
; RUN: not llvm-link %p/Inputs/appending-global.ll %s -S -o - 2>&1 | FileCheck %s

; Negative test to check that global variables with appending linkage and
; different constness cannot be linked.

; CHECK: Appending variables linked with different const'ness

@var = appending constant [1 x i8* ] undef
