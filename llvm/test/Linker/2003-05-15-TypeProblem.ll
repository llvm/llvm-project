; This one fails because the LLVM runtime is allowing two null pointers of
; the same type to be created!

; RUN: echo "%M = type { %N*} %N = type opaque" | llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc

%M = type { i32* }
%N = type i32

