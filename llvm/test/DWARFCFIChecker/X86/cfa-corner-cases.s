# RUN: not llvm-mc %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s --allow-empty
.text
        .globl  f
        .type   f, @function
f:
        .cfi_startproc
        
        # TODO: Remove these lines when the initial frame directives set the callee saved registers
        .cfi_undefined %rax
        .cfi_undefined %flags

        pushq   %rax
        # CHECK: error: modified CFA register RSP but not changed CFA rule
        
        addq    $10, %rax
        # CHECK: error: did not modify CFA register RSP but changed CFA rule
        .cfi_adjust_cfa_offset 8

.Lfunc_end0:
        .size   f, .Lfunc_end0-f
        .cfi_endproc
