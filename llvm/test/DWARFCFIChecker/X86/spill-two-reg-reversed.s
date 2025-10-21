# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s 
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .type   _start,@function
        .globl  _start
        .hidden _start
_start:
        .cfi_startproc

        .cfi_same_value %rdi
        .cfi_same_value %rsi

        pushq   %rbp
        .cfi_adjust_cfa_offset 8
        .cfi_offset %rbp, -16

        movq    %rsp, %rbp

        pushq   %rdi
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rdi, 0

        pushq   %rsi
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rsi, 0
        
        popq    %rsi
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rdi

        popq    %rdi
        # CHECK: error: changed register RDI, that register RDI's unwinding rule uses, but there is no CFI directives about it
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rsi

        popq    %rbp
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rbp

        retq

        .cfi_endproc
.Ltmp0:
        .size   _start, .Ltmp0-_start
        .text
