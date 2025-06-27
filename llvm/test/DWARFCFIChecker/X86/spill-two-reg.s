# RUN: llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s 
# TODO: Should check no warnings are emitted but for now, the tool is naive and emitting warnings for every change.
        .text
        .type   _start,@function
        .globl  _start
        .hidden _start
_start:
        .cfi_startproc

        .cfi_same_value %rdi
        .cfi_same_value %rsi

        pushq   %rbp
        # CHECK: warning: unknown change happened to register RBP unwinding rule structure
        .cfi_adjust_cfa_offset 8
        .cfi_offset %rbp, -16

        movq    %rsp, %rbp

        pushq   %rdi
        # CHECK: warning: unknown change happened to register RDI unwinding rule structure
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rdi, 0

        pushq   %rsi
        # CHECK: warning: unknown change happened to register RSI unwinding rule structure
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rsi, 0
        
        popq    %rsi
        # CHECK: warning: unknown change happened to register RSI unwinding rule structure
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rsi

        popq    %rdi
        # CHECK: warning: unknown change happened to register RDI unwinding rule structure
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rdi

        popq    %rbp
        # CHECK: warning: unknown change happened to register RBP unwinding rule structure
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rbp

        retq

        .cfi_endproc
.Ltmp0:
        .size   _start, .Ltmp0-_start
        .text
