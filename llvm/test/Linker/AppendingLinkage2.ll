; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo "@X = appending global [1 x i32] [i32 8] " | \
; RUN:   llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | grep 7 | grep 8

@X = appending global [1 x i32] [ i32 7 ]		; <[1 x i32]*> [#uses=0]
