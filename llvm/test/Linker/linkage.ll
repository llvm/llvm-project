; RUN: llvm-as %S/Inputs/linkage.a.ll -o %t.1.bc
; RUN: llvm-as %S/Inputs/linkage.b.ll -o %t.2.bc
; RUN: llvm-link %t.1.bc  %t.2.bc
