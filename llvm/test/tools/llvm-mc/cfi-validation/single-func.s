# RUN: llvm-mc %s --validate-cfi --filetype=null
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc
        
        .cfi_undefined %rax
        
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        
        movl    %edi, -4(%rbp)
        
        movl    -4(%rbp), %eax
        
        addl    $10, %eax
        
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        
        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
