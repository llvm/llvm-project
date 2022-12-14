; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%M = type { i32, ptr } " | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc

%M = type { i32, ptr }
%N = type opaque

;%X = global { int, ptr } { int 5, ptr null }
