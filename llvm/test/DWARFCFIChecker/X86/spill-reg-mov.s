# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --allow-empty --implicit-check-not warning: --implicit-check-not error:
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        .cfi_same_value %rbx
        .cfi_same_value %r12

        ## Spill with plain stores. CFA is %rsp + 8 here, so -16(%rsp) is
        ## CFA - 24 and -8(%rsp) is CFA - 16.
        movq    %rbx, -16(%rsp)
        .cfi_offset %rbx, -24
        movq    %r12, -8(%rsp)
        .cfi_offset %r12, -16

        ## Restore with plain loads from the same slots.
        movq    -16(%rsp), %rbx
        .cfi_same_value %rbx
        movq    -8(%rsp), %r12
        .cfi_same_value %r12

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
