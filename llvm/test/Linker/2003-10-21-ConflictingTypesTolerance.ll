; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo { %S = type \[8 x i32\] external global %S } | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out1.bc %t.out2.bc | llvm-dis | grep %S | grep \\{
%S = type { i32 }


