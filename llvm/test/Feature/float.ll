; RUN: llvm-as < %s | llvm-dis > t1.ll
; RUN: llvm-as t1.ll -o - | llvm-dis > t2.ll
; RUN: diff t1.ll t2.ll

@F1     = global float 0x4010000000000000
@D1     = global double 0x4010000000000000
