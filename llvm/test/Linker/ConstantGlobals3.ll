; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo {@X = external constant \[1 x i32\] } | \
; RUN:   llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc | llvm-dis | grep constant

@X = external global [1 x i32]		; <[1 x i32]*> [#uses=0]
