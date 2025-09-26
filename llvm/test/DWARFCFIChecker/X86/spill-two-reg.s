# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s 
## TODO: Should check no warnings are emitted but for now, the tool is naive and emitting warnings for every change.
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
        # CHECK: warning: CFA offset is changed from 8 to 16, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK: warning: validating changes happening to register RBP unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset 8
        .cfi_offset %rbp, -16

        movq    %rsp, %rbp

        pushq   %rdi
        # CHECK: warning: CFA offset is changed from 16 to 24, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK: warning: validating changes happening to register RDI unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rdi, 0

        pushq   %rsi
        # CHECK: warning: CFA offset is changed from 24 to 32, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK: warning: validating changes happening to register RSI unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset %rsi, 0
        
        popq    %rsi
        # CHECK: warning: CFA offset is changed from 32 to 24, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK: warning: validating changes happening to register RSI unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rsi

        popq    %rdi
        # CHECK: warning: CFA offset is changed from 24 to 16, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK: warning: validating changes happening to register RDI unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rdi

        popq    %rbp
        # CHECK: warning: CFA offset is changed from 16 to 8, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK: warning: validating changes happening to register RBP unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rbp

        retq

        .cfi_endproc
.Ltmp0:
        .size   _start, .Ltmp0-_start
        .text
