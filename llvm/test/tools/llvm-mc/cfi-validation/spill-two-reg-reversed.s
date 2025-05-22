# RUN: not llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s 
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
        # CHECK: warning: The reg#55 CFI state is changed
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rdi

        popq    %rdi
        # CHECK: warning: The reg#60 CFI state is changed
        # CHECK: Reg#55 caller's value is in reg#55 which is changed by this instruction, but not changed in CFI directives
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
