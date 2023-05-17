; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo "@X = appending global [1 x i32] [i32 8] " | \
; RUN:   llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | FileCheck %s
; CHECK: [i32 7, i32 8]

@X = appending global [1 x i32] [ i32 7 ]		; <ptr> [#uses=0]
