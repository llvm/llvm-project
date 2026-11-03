# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --implicit-check-not warning: --implicit-check-not error:
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        .cfi_same_value %rbx

        ## -16(%rsp) is CFA - 24, not CFA - 16.
        movq    %rbx, -16(%rsp)
        # CHECK: error: expected register RBX unwinding rule's offset to be -24, but the CFI directives set it to -16
        .cfi_offset %rbx, -16

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
