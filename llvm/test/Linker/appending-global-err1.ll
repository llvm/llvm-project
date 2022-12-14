; RUN: not llvm-link %s %p/Inputs/appending-global.ll -S -o - 2>&1 | FileCheck %s
; RUN: not llvm-link %p/Inputs/appending-global.ll %s -S -o - 2>&1 | FileCheck %s

; Negative test to check that global variable with appending linkage can only be
; linked with another global variable with appending linkage.

; CHECK: can only link appending global with another appending global

@var = global ptr undef
