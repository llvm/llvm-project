; RUN: llvm-as < %s > %t.bc
; RUN: echo | llvm-as > %t.tmp.bc
; RUN: llvm-link %t.tmp.bc %t.bc

@X = constant i32 5		; <i32*> [#uses=2]
@Y = internal global [2 x i32*] [ i32* @X, i32* @X ]		; <[2 x i32*]*> [#uses=0]


