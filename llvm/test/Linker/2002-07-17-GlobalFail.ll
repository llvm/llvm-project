; RUN: llvm-as < %s > %t.bc
; RUN: echo | llvm-as > %t.tmp.bc
; RUN: llvm-link %t.tmp.bc %t.bc

@X = constant i32 5		; <ptr> [#uses=2]
@Y = internal global [2 x ptr] [ ptr @X, ptr @X ]		; <ptr> [#uses=0]


