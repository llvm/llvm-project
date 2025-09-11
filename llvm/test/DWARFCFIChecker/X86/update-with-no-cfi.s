# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s 
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
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
        # CHECK: error: changed register RAX, that register RAX's unwinding rule uses, but there is no CFI directives about it

        movq $10, %rbx
        # CHECK: error: changed register RBX, that register RBX's unwinding rule uses, but there is no CFI directives about it

        movq $10, %rcx
        # CHECK: error: changed register RCX, that register RCX's unwinding rule uses, but there is no CFI directives about it

        movq $10, %rdx
        # CHECK: error: changed register RDX, that register RDX's unwinding rule uses, but there is no CFI directives about it

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
