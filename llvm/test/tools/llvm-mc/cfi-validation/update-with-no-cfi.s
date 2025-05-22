# RUN: not llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s 
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        .cfi_same_value %rax
        .cfi_same_value %rbx
        .cfi_same_value %rcx
        .cfi_same_value %rdx

        movq $10, %rax
        # CHECK: error: Reg#51 caller's value is in reg#51 which is changed by this instruction, but not changed in CFI directives

        movq $10, %rbx
        # CHECK: error: Reg#53 caller's value is in reg#53 which is changed by this instruction, but not changed in CFI directives

        movq $10, %rcx
        # CHECK: error: Reg#54 caller's value is in reg#54 which is changed by this instruction, but not changed in CFI directives

        movq $10, %rdx
        # CHECK: error: Reg#56 caller's value is in reg#56 which is changed by this instruction, but not changed in CFI directives

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
