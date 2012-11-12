; RUN: echo "%%T = type opaque" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc

%T = type opaque
@a = constant { %T* } zeroinitializer		; <{ %T* }*> [#uses=0]

