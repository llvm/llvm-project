# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
        
        ## TODO: Remove these lines when the initial frame directives set the callee saved registers
        .cfi_undefined %rax
        .cfi_undefined %flags

        .cfi_val_offset %rbp, -8 

        pushq   %rax
        # CHECK: error: modified CFA register RSP but not changed CFA rule
        
        addq    $10, %rax
        # CHECK: error: did not modify CFA register RSP but changed CFA rule
        .cfi_adjust_cfa_offset 8

.Lfunc_end0:
        .size   f, .Lfunc_end0-f
        .cfi_endproc
