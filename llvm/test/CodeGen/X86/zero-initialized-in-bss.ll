; RUN: llc < %s -mtriple=x86_64 | FileCheck %s --check-prefix=BSS

; BSS:      .bss
; BSS-NEXT: .globl a
; BSS:      .section .tbss,"awT",@nobits
; BSS-NEXT: .globl b

; RUN: llc < %s -mtriple=x86_64 -nozero-initialized-in-bss | FileCheck %s --check-prefix=DATA

; DATA:      .data
; DATA-NEXT: .globl a
; DATA:      .section .tdata,"awT",@progbits
; DATA-NEXT: .globl b

@a = global i32 0
@b = thread_local global i32 0
