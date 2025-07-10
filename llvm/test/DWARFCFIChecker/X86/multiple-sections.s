# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s 
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
        pushq %rax
        # CHECK: error: modified CFA register RSP but not changed CFA rule
        pushq %rax
        .cfi_def_cfa %rbp, -24
        pushq %rax
        .cfi_endproc

        .cfi_startproc
        pushq %rax
        # CHECK: error: modified CFA register RSP but not changed CFA rule
        pushq %rax
        .cfi_def_cfa %rbp, -24
        pushq %rax
        .cfi_endproc

        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f
