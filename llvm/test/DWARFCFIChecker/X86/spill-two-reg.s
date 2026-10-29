# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --implicit-check-not warning: --implicit-check-not error:
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
        # CHECK: warning: validating changes happening to register RBP unwinding rule structure is not implemented yet
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
        .cfi_same_value %rsi

        popq    %rdi
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rdi

        popq    %rbp
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rbp

        retq

        .cfi_endproc
.Ltmp0:
        .size   _start, .Ltmp0-_start
        .text
