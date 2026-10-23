# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --allow-empty --implicit-check-not warning: --implicit-check-not error:
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        .cfi_same_value %rbx
        .cfi_undefined %rax

        ## Keep the caller's %rbx in %rax.
        movq    %rbx, %rax
        .cfi_register %rbx, %rax

        movq    $10, %rbx

        ## Put it back.
        movq    %rax, %rbx
        .cfi_same_value %rbx

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
