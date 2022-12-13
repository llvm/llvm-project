; Test that a prototype can be marked const, and the definition is allowed
; to be nonconst.

; RUN: echo "@X = global i32 7" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | FileCheck %s
; CHECK: global i32 7

@X = external constant i32		; <ptr> [#uses=0]
