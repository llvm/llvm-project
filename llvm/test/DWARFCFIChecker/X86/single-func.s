# RUN: llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s 
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .text
        .text
        .globl  f
        .type   f,@function
f:
        .cfi_startproc
        
        ## TODO: Remove these lines when the initial frame directives set the callee saved registers
        .cfi_undefined %rax
        .cfi_undefined %flags
        
        pushq   %rbp
        # CHECK: warning: CFA offset is changed from 8 to 16, and CFA register RSP is modified, but validating the modification amount is not implemented yet
        # CHECK:  warning: validating changes happening to register RBP unwinding rule structure is not implemented yet
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        
        movq    %rsp, %rbp
        # CHECK:  warning: CFA register changed from register RSP to register RBP, validating this change is not implemented yet
        .cfi_def_cfa_register %rbp
        
        movl    %edi, -4(%rbp)
        
        movl    -4(%rbp), %eax
        
        addl    $10, %eax
        
        popq    %rbp
        # CHECK:  warning: CFA register changed from register RBP to register RSP, validating this change is not implemented yet
        .cfi_def_cfa %rsp, 8
        
        retq

.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .cfi_endproc
