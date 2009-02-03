; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo {%S = type \{ i32, i32* \} } | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc

%S = type { i32, %T* }
%T = type opaque

;%X = global { int, %T* } { int 5, %T* null }
