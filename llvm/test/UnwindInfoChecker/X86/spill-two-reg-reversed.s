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
        # CHECK: warning: Unknown change happened to %RBP unwinding rule
        .cfi_adjust_cfa_offset 8
        .cfi_offset %rbp, -16

        movq    %rsp, %rbp

        pushq   %rdi
        # CHECK: warning: Unknown change happened to %RDI unwinding rule
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rdi, 0

        pushq   %rsi
        # CHECK: warning: Unknown change happened to %RSI unwinding rule
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rsi, 0
        
        popq    %rsi
        # CHECK: warning: Unknown change happened to %RDI unwinding rule
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rdi

        popq    %rdi
        # CHECK: error: This instruction changes %RDI, that %RDI unwinding rule uses, but there is no CFI directives about it
        # CHECK: warning: Unknown change happened to %RSI unwinding rule
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rsi

        popq    %rbp
        # CHECK: warning: Unknown change happened to %RBP unwinding rule
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rbp

        retq

        .cfi_endproc
.Ltmp0:
        .size   _start, .Ltmp0-_start
        .text
