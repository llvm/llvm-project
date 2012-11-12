; RUN: llc < %s -march=cellspu | FileCheck %s

@bssVar = global i32 zeroinitializer
; CHECK: .section .bss
; CHECK-NEXT: .globl

@localVar= internal global i32 zeroinitializer
; CHECK-NOT: .lcomm
; CHECK: .local
; CHECK-NEXT: .comm

