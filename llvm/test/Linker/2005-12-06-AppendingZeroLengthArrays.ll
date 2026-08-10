; RUN: echo " @G = appending global [0 x i32] zeroinitializer " | \
; RUN:   llvm-as > %t.out2.bc
; RUN: llvm-as < %s > %t.out1.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc -S | FileCheck %s
; CHECK: @G =

; When linked, the globals should be merged, and the result should still 
; be named '@G'.

@G = appending global [1 x i32] zeroinitializer		; <ptr> [#uses=0]

