# RUN: not llvm-mc -triple x86_64-pc-linux-gnu %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s 
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
