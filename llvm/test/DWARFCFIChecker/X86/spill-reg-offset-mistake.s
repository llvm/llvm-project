# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --implicit-check-not warning: --implicit-check-not error:
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        .cfi_same_value %rbx

        ## push stores %rbx at %rsp - 8, which is CFA - 16.
        pushq   %rbx
        # CHECK: error: expected register RBX unwinding rule's offset to be -16, but the CFI directives set it to -24
        .cfi_adjust_cfa_offset 8
        .cfi_offset %rbx, -24

        ## pop reads %rbx back from CFA - 16, but the directives said it was
        ## saved at CFA - 24, so the restore cannot be validated.
        popq    %rbx
        # CHECK: warning: validating changes happening to register RBX unwinding rule structure is not implemented yet
        .cfi_adjust_cfa_offset -8
        .cfi_same_value %rbx

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
