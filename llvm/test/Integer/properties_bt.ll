; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


target datalayout = "e-p:32:32"
target triple = "proc-vend-sys"
deplibs = [ "m", "c" ]

