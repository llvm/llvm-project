; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1
; REQUIRES: default_triple

module asm ".align 4"
