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
        # CHECK: error: This instruction changes %RAX, that %RAX unwinding rule uses, but there is no CFI directives about it

        movq $10, %rbx
        # CHECK: error: This instruction changes %RBX, that %RBX unwinding rule uses, but there is no CFI directives about it

        movq $10, %rcx
        # CHECK: error: This instruction changes %RCX, that %RCX unwinding rule uses, but there is no CFI directives about it

        movq $10, %rdx
        # CHECK: error: This instruction changes %RDX, that %RDX unwinding rule uses, but there is no CFI directives about it

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
