# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s --implicit-check-not warning: --implicit-check-not error:
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc

        pushq   %rbp
        # CHECK: error: expected CFA offset 16 from CFA register RSP, but the CFI directives set it to 17
        # CHECK: warning: validating changes happening to register RBP unwinding rule structure is not implemented yet
        .cfi_def_cfa_offset 17
        .cfi_offset %rbp, -16

        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp

        movl    %edi, -4(%rbp)

        movl    -4(%rbp), %eax

        addl    $10, %eax

        popq    %rbp
        # CHECK: warning: CFA register changed from register RBP to register RSP, validating this change is not implemented yet
        .cfi_def_cfa %rsp, 8

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
