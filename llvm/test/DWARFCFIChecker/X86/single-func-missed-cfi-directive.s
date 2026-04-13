# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s 
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc
        
        .cfi_undefined %rax
        
        pushq   %rbp
        .cfi_def_cfa_offset 16
        
        movq    %rsp, %rbp
        # CHECK: error: changed register RBP, that register RBP's unwinding rule uses, but there is no CFI directives about it
        .cfi_def_cfa_register %rbp
        
        movl    %edi, -4(%rbp)
        
        movl    -4(%rbp), %eax
        
        addl    $10, %eax
        
        popq    %rbp
        # CHECK: error: changed register RBP, that register RBP's unwinding rule uses, but there is no CFI directives about it
        .cfi_def_cfa %rsp, 8
        
        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
