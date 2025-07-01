# RUN: llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s 
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        # TODO: Remove these lines when the initial frame directives set the callee saved registers
        .cfi_undefined %rax
        .cfi_undefined %flags

        pushq   %rbp
        # CHECK: warning: CFA offset is changed from 8 to 17, CFA register RSP is changed by an unknown amount
        # CHECK: warning: uncheckable change happened to register RBP unwinding rule structure
        .cfi_def_cfa_offset 17
        .cfi_offset %rbp, -16

        movq    %rsp, %rbp
        # CHECK: warning: CFA register changed from register RSP to register RBP
        .cfi_def_cfa_register %rbp

        movl    %edi, -4(%rbp)

        movl    -4(%rbp), %eax

        addl    $10, %eax

        popq    %rbp
        # CHECK: warning: CFA register changed from register RBP to register RSP
        .cfi_def_cfa %rsp, 8

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
